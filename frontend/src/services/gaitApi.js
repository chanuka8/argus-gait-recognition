
import { getApiUrl, getWsUrl, getStreamUrl, getSnapshotUrl } from '../config/apiConfig';

async function request(endpoint, options = {}) {
  const url = getApiUrl(endpoint);
  try {
    const response = await fetch(url, options);
    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP Error ${response.status}: ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`[gaitApi] Request failed for ${endpoint}:`, error);
    if (error instanceof TypeError && error.message.toLowerCase().includes('fetch')) {
      throw new Error('Unable to reach ARGUS backend server (is the API server running on port 8000?)');
    }
    throw error;
  }
}

export function normalizeRecognitionEvent(rawEvent) {
  if (!rawEvent || typeof rawEvent !== 'object') return null;

  const identity = rawEvent.identity || rawEvent.person_id || 'UNKNOWN';
  const confidence = typeof rawEvent.confidence === 'number'
    ? rawEvent.confidence
    : typeof rawEvent.similarity === 'number'
      ? rawEvent.similarity
      : 0.0;

  let decision = rawEvent.decision;
  if (!decision) {
    if (rawEvent.status === 'CONFIRMED' || identity !== 'UNKNOWN') {
      decision = 'KNOWN';
    } else {
      decision = 'UNKNOWN';
    }
  }

  return {
    event_id: rawEvent.event_id || `evt-${Date.now()}-${Math.random().toString(36).substr(2, 6)}`,
    camera_id: rawEvent.camera_id || 'upload-image',
    track_id: rawEvent.track_id != null ? rawEvent.track_id : 'N/A',
    identity,
    person_id: identity,
    confidence: Number(confidence) || 0.0,
    similarity: Number(confidence) || 0.0,
    decision: String(decision).toUpperCase(),
    status: rawEvent.status || (decision === 'KNOWN' ? 'CONFIRMED' : 'UNKNOWN'),
    recognition_branch: rawEvent.recognition_branch || '2D_GEI',
    timestamp: rawEvent.timestamp || new Date().toISOString(),
    bbox: Array.isArray(rawEvent.bbox) ? rawEvent.bbox : null,
    quality: rawEvent.quality != null ? Number(rawEvent.quality) : 0.85,
  };
}

export const gaitApi = {
  getHealth: () => request('/health'),
  getStatus: () => request('/status'),
  getMetrics: () => request('/metrics'),
  getEvents: () => request('/events'),
  getCameras: () => request('/cameras'),
  getCameraInfo: (cameraId) => request(`/cameras/${encodeURIComponent(cameraId)}`),

  getStreamUrl,
  getSnapshotUrl,

  identifyImage: async (file, cameraId = 'upload-image') => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('camera_id', cameraId);
    const raw = await request('/identify/image', {
      method: 'POST',
      body: formData,
    });
    return normalizeRecognitionEvent(raw);
  },

  analyzeVideo: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const rawList = await request('/analyze/video', {
      method: 'POST',
      body: formData,
    });
    if (Array.isArray(rawList)) {
      return rawList.map(normalizeRecognitionEvent).filter(Boolean);
    }
    return [normalizeRecognitionEvent(rawList)].filter(Boolean);
  },

  startCamera: async (cameraId, source, location = 'Surveillance Zone', zoneId = null) => {
    const payload = { camera_id: cameraId, source, location };
    if (zoneId) payload.zone_id = zoneId;
    return request('/cameras/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  },

  stopCamera: async (cameraId) => {
    return request('/cameras/stop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ camera_id: cameraId }),
    });
  },

  enrollSubject: async (personId, files) => {
    const formData = new FormData();
    formData.append('person_id', personId);
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }
    return request('/enroll', {
      method: 'POST',
      body: formData,
    });
  },

  createWebSocket: (onEvent, onError, onClose) => {
    const wsUrl = getWsUrl('/ws/events');
    let socket = null;
    let reconnectAttempts = 0;
    const maxReconnectAttempts = 8;
    let isExplicitClosed = false;
    let reconnectTimer = null;

    const connect = () => {
      if (isExplicitClosed) return;

      try {
        socket = new WebSocket(wsUrl);

        socket.onopen = () => {
          reconnectAttempts = 0;
          console.log('[gaitApi] WebSocket connected to ARGUS Engine:', wsUrl);
        };

        socket.onmessage = (evt) => {
          try {
            const data = JSON.parse(evt.data);
            const normalized = normalizeRecognitionEvent(data);
            if (normalized && onEvent) onEvent(normalized);
          } catch (err) {
            console.error('[gaitApi] Failed to parse WS message:', err);
          }
        };

        socket.onerror = (err) => {
          if (onError) onError(err);
        };

        socket.onclose = () => {
          if (onClose) onClose();
          if (!isExplicitClosed && reconnectAttempts < maxReconnectAttempts) {
            reconnectAttempts++;
            const delay = Math.min(1000 * Math.pow(1.5, reconnectAttempts), 10000);
            console.warn(`[gaitApi] WS disconnected. Reconnecting in ${Math.round(delay)}ms (attempt ${reconnectAttempts})...`);
            clearTimeout(reconnectTimer);
            reconnectTimer = setTimeout(connect, delay);
          }
        };
      } catch (err) {
        if (onError) onError(err);
      }
    };

    connect();

    return {
      close: () => {
        isExplicitClosed = true;
        clearTimeout(reconnectTimer);
        if (socket) {
          try {
            socket.close();
          } catch {
            /* ignore */
          }
        }
      },
    };
  },
};
