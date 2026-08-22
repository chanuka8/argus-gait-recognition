import React from 'react';
import { useGait } from '../contexts/GaitContext';
import './GaitSystemStatus.css';

export const GaitSystemStatus = () => {
  const { health, status, metrics, isConnected, loading, error } = useGait();

  if (loading && !health) {
    return (
      <div className="gait-status-widget">
        <div style={{ color: 'var(--text-muted)', fontSize: '0.88rem' }}>Loading ARGUS Gait Engine Telemetry...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="gait-status-widget">
        <div className="gait-status-error">
          <span>⚠️ Gait Engine Offline: {error}</span>
          <span className="gait-pill degraded">OFFLINE</span>
        </div>
      </div>
    );
  }

  const isHealthy = health?.status === 'healthy';

  return (
    <div className="gait-status-widget">
      <div className="gait-status-header">
        <div className="gait-status-title-group">
          <span className={`status-pulse-dot ${isHealthy ? '' : 'offline'}`} />
          <h3>ARGUS Gait Engine</h3>
        </div>
        <div className="gait-status-badges">
          <span className={`gait-pill ${isConnected ? 'ws-live' : 'ws-off'}`}>
            {isConnected ? '⚡ WS Live' : 'Polling'}
          </span>
          <span className={`gait-pill ${isHealthy ? 'healthy' : 'degraded'}`}>
            {isHealthy ? 'HEALTHY' : 'DEGRADED'}
          </span>
        </div>
      </div>

      <div className="gait-metrics-grid">
        <div className="gait-metric-box">
          <div className="gait-metric-label">Device</div>
          <div className="gait-metric-val">{status?.device?.toUpperCase() || 'CPU'}</div>
        </div>

        <div className="gait-metric-box">
          <div className="gait-metric-label">Identities</div>
          <div className="gait-metric-val" style={{ color: '#06D6A0' }}>{metrics?.people || 0}</div>
        </div>

        <div className="gait-metric-box">
          <div className="gait-metric-label">Embeddings</div>
          <div className="gait-metric-val" style={{ color: '#FFD166' }}>{metrics?.embeddings || 0}</div>
        </div>

        <div className="gait-metric-box">
          <div className="gait-metric-label">Active Cameras</div>
          <div className="gait-metric-val" style={{ color: '#5CE1E6' }}>{status?.active_cameras || 0}</div>
        </div>
      </div>
    </div>
  );
};

export default GaitSystemStatus;
