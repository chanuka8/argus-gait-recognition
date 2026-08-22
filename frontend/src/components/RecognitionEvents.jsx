import React, { useState } from 'react';
import { useGait } from '../contexts/GaitContext';
import './RecognitionEvents.css';

export const RecognitionEvents = () => {
  const { events } = useGait();
  const [filter, setFilter] = useState('ALL');

  const filteredEvents = events.filter((evt) => {
    if (filter === 'ALL') return true;
    const decision = evt.decision || (evt.identity !== 'UNKNOWN' ? 'KNOWN' : 'UNKNOWN');
    return decision === filter;
  });

  const getDecisionBadge = (decision, identity) => {
    const dec = decision || (identity && identity !== 'UNKNOWN' ? 'KNOWN' : 'UNKNOWN');
    switch (dec) {
      case 'KNOWN':
      case 'CONFIRMED':
        return <span className="decision-badge known">KNOWN</span>;
      case 'UNCERTAIN':
        return <span className="decision-badge uncertain">UNCERTAIN</span>;
      default:
        return <span className="decision-badge unknown">UNKNOWN</span>;
    }
  };

  const formatConfidence = (conf, sim) => {
    const val = typeof conf === 'number' ? conf : typeof sim === 'number' ? sim : 0;
    return `${(val * 100).toFixed(1)}% Match`;
  };

  const formatTime = (ts) => {
    if (!ts) return '';
    try {
      return new Date(ts).toLocaleTimeString();
    } catch {
      return String(ts);
    }
  };

  return (
    <div className="recon-events-widget">
      <div className="recon-events-header">
        <div className="recon-events-title">
          <h3>📡 Real-Time Recognition Stream</h3>
          <span className="recon-count-tag">{events.length} Events</span>
        </div>

        <div className="recon-filter-group">
          {['ALL', 'KNOWN', 'UNCERTAIN', 'UNKNOWN'].map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`recon-filter-btn ${filter === f ? 'active' : ''}`}
            >
              {f}
            </button>
          ))}
        </div>
      </div>

      {filteredEvents.length === 0 ? (
        <div className="recon-empty-state">No recognition events matching filter.</div>
      ) : (
        <div className="recon-events-list">
          {filteredEvents.map((evt) => {
            const identity = evt.identity || evt.person_id || 'UNKNOWN';
            return (
              <div key={evt.event_id || `${identity}-${evt.timestamp}`} className="recon-event-card">
                <div className="recon-event-left">
                  {getDecisionBadge(evt.decision, identity)}
                  <div>
                    <div className="recon-identity">{identity}</div>
                    <div className="recon-meta">
                      Cam: {evt.camera_id || 'stream'} | Track: #{evt.track_id ?? 'N/A'} | {evt.recognition_branch || '2D_GEI'}
                    </div>
                  </div>
                </div>

                <div className="recon-event-right">
                  <div className="recon-conf">
                    {formatConfidence(evt.confidence, evt.similarity)}
                  </div>
                  <div className="recon-time">
                    {formatTime(evt.timestamp)}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default RecognitionEvents;
