
const envApiUrl = import.meta.env.VITE_GAIT_API_URL || import.meta.env.VITE_API_URL;
const isDev = import.meta.env.DEV;

export const API_BASE = envApiUrl !== undefined
  ? envApiUrl
  : (isDev ? 'http://localhost:8000' : '');

export const getApiUrl = (endpoint = '') => {
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  return `${API_BASE}/api/v1${cleanEndpoint}`;
};

export const getStreamUrl = (cameraId) => {
  return `${API_BASE}/api/v1/cameras/${encodeURIComponent(cameraId)}/stream`;
};

export const getSnapshotUrl = (cameraId) => {
  return `${API_BASE}/api/v1/cameras/${encodeURIComponent(cameraId)}/snapshot`;
};

export const getWsUrl = (wsPath = '/ws/events') => {
  const cleanPath = wsPath.startsWith('/') ? wsPath : `/${wsPath}`;
  if (API_BASE.startsWith('http://') || API_BASE.startsWith('https://')) {
    return API_BASE.replace(/^http/, 'ws') + `/api/v1${cleanPath}`;
  }
  if (typeof window !== 'undefined') {
    const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${proto}//${window.location.host}/api/v1${cleanPath}`;
  }
  return `ws://localhost:8000/api/v1${cleanPath}`;
};

export const FACE_API_BASE_URL = import.meta.env.VITE_FACE_API_URL || (isDev ? 'http://localhost:8000' : '');
