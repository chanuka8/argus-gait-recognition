import React, { useState, useEffect, useCallback } from 'react';
import { gaitApi, normalizeRecognitionEvent } from '../services/gaitApi';
import { getStreamUrl, getSnapshotUrl } from '../config/apiConfig';
import { GaitContext } from './gaitContextDef';
import { useAuth } from '../hooks/useAuth';

export const GaitProvider = ({ children }) => {
  const { currentUser } = useAuth();
  const [health, setHealth] = useState(null);
  const [status, setStatus] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [events, setEvents] = useState([]);
  const [cameras, setCameras] = useState([]);
  const [isConnected, setIsConnected] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchState = useCallback(async () => {
    try {
      setLoading(true);
      const token = sessionStorage.getItem('argus_session_token');
      const queries = [gaitApi.getHealth(), gaitApi.getStatus()];
      if (token) {
        queries.push(gaitApi.getMetrics(), gaitApi.getEvents(), gaitApi.getCameras());
      }
      const results = await Promise.allSettled(queries);
      const [healthRes, statusRes] = results;
      const metricsRes = token ? results[2] : null;
      const eventsRes = token ? results[3] : null;
      const camsRes = token ? results[4] : null;

      if (healthRes?.status === 'fulfilled') setHealth(healthRes.value);
      if (statusRes?.status === 'fulfilled') setStatus(statusRes.value);
      if (metricsRes?.status === 'fulfilled') setMetrics(metricsRes.value);
      if (eventsRes?.status === 'fulfilled') {
        const rawEvents = Array.isArray(eventsRes.value) ? eventsRes.value : [];
        setEvents(rawEvents.map(normalizeRecognitionEvent).filter(Boolean));
      }
      if (camsRes?.status === 'fulfilled') {
        setCameras(Array.isArray(camsRes.value) ? camsRes.value : []);
      }

      setError(null);
    } catch (err) {
      console.error('[GaitContext] Error fetching gait status:', err);
      setError('Unable to connect to ARGUS Gait Engine');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchState();
    const interval = setInterval(fetchState, 10000);

    const ws = gaitApi.createWebSocket(
      (newEvent) => {
        setIsConnected(true);
        if (newEvent) {
          setEvents((prev) => {
            const exists = prev.some((e) => e.event_id === newEvent.event_id);
            if (exists) return prev;
            return [newEvent, ...prev.slice(0, 99)];
          });
        }
      },
      () => setIsConnected(false),
      () => setIsConnected(false)
    );

    return () => {
      clearInterval(interval);
      ws.close();
    };
  }, [fetchState, currentUser]);

  const identifyImage = async (file, cameraId) => {
    const event = await gaitApi.identifyImage(file, cameraId);
    if (event) {
      setEvents((prev) => [event, ...prev.slice(0, 99)]);
    }
    fetchState();
    return event;
  };

  const analyzeVideo = async (file) => {
    const newEvents = await gaitApi.analyzeVideo(file);
    if (Array.isArray(newEvents) && newEvents.length > 0) {
      setEvents((prev) => [...newEvents, ...prev].slice(0, 100));
    }
    fetchState();
    return newEvents;
  };

  const startCamera = async (cameraId, source, location, zoneId) => {
    const res = await gaitApi.startCamera(cameraId, source, location, zoneId);
    if (res && res.camera_id) {
      setCameras((prev) => {
        const filtered = prev.filter((c) => c.camera_id !== res.camera_id);
        return [...filtered, res];
      });
    }
    fetchState();
    return res;
  };

  const stopCamera = async (cameraId) => {
    const res = await gaitApi.stopCamera(cameraId);
    setCameras((prev) => prev.filter((c) => c.camera_id !== cameraId));
    fetchState();
    return res;
  };

  const getCameraInfo = async (cameraId) => {
    return gaitApi.getCameraInfo(cameraId);
  };

  const enrollSubject = async (personId, files) => {
    const res = await gaitApi.enrollSubject(personId, files);
    fetchState();
    return res;
  };

  const value = {
    health,
    status,
    metrics,
    events,
    cameras,
    isConnected,
    loading,
    error,
    refreshState: fetchState,
    identifyImage,
    analyzeVideo,
    startCamera,
    stopCamera,
    getCameraInfo,
    getStreamUrl,
    getSnapshotUrl,
    getCameraSnapshot: gaitApi.getCameraSnapshot,
    enrollSubject,
  };

  return <GaitContext.Provider value={value}>{children}</GaitContext.Provider>;
};

export default GaitProvider;
