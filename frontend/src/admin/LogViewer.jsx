import React, { useState, useEffect, useCallback } from 'react';
import { Search, Filter, Download, AlertTriangle, Info, Skull, Trash2, X, ExternalLink } from 'lucide-react';
import AdminHeader from './AdminHeader';
import { getLogs, clearLogs, exportLogsCSV } from '../utils/logService';
import './LogViewer.css';

const LogViewer = () => {
    const [logs, setLogs] = useState(() => getLogs());
    const [searchTerm, setSearchTerm] = useState('');
    const [selectedLevel, setSelectedLevel] = useState('all');
    const [selectedLog, setSelectedLog] = useState(null);

    // Load logs from the centralized service
    const refreshLogs = useCallback(() => {
        setLogs(getLogs());
    }, []);

    // Listen for real-time log events from other components
    useEffect(() => {
        const handleLogUpdate = () => {
            refreshLogs();
        };

        window.addEventListener('argus-log-update', handleLogUpdate);
        return () => window.removeEventListener('argus-log-update', handleLogUpdate);
    }, [refreshLogs]);

    const handleClearLogs = () => {
        if (window.confirm('Are you sure you want to clear all system logs?')) {
            clearLogs();
            setLogs([]);
        }
    };

    const handleExport = () => {
        exportLogsCSV();
    };

    const getLevelIcon = (level) => {
        switch (level) {
            case 'critical': return <Skull size={14} color="var(--status-missing)" />;
            case 'warning': return <AlertTriangle size={14} color="var(--status-investigating)" />;
            default: return <Info size={14} color="var(--sky)" />;
        }
    };

    const filteredLogs = logs.filter(log => {
        const matchesSearch = log.message.toLowerCase().includes(searchTerm.toLowerCase()) ||
            log.user.toLowerCase().includes(searchTerm.toLowerCase());
        const matchesLevel = selectedLevel === 'all' || log.level === selectedLevel;
        return matchesSearch && matchesLevel;
    });

    return (
        <div className="log-viewer-page">
            <AdminHeader />

            <main className="log-viewer-content">
                <div className="log-header-row">
                    <div className="title-group">
                        <h1>System Logs</h1>
                    </div>
                    <div className="log-actions">
                        <button className="log-action-btn export" onClick={handleExport} title="Export to CSV">
                            <Download size={16} />
                            <span>Export CSV</span>
                        </button>
                        <button className="log-action-btn purge" onClick={handleClearLogs} title="Clear all logs">
                            <Trash2 size={16} />
                            <span>Clear Logs</span>
                        </button>
                    </div>
                </div>

                <div className="log-filter-controls">
                    <div className="log-search-bar">
                        <Search size={18} color="var(--text-muted)" />
                        <input
                            type="text"
                            placeholder="Search logs by keyword, user or event details..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                        />
                    </div>

                    <div className="filters-row">
                        <div className="filter-select-group">
                            <Filter size={14} color="var(--text-muted)" />
                            <label>Severity:</label>
                            <select value={selectedLevel} onChange={(e) => setSelectedLevel(e.target.value)}>
                                <option value="all">ALL LEVELS</option>
                                <option value="info">INFO</option>
                                <option value="warning">WARNING</option>
                                <option value="critical">CRITICAL</option>
                            </select>
                        </div>
                    </div>
                </div>

                <div className="logs-terminal-container">
                    <div className="terminal-header">
                        <div className="terminal-dots">
                            <span className="dot red"></span>
                            <span className="dot yellow"></span>
                            <span className="dot green"></span>
                        </div>
                        <span className="terminal-title">argus_audit_stream.log</span>
                        <span className="terminal-count">Showing {filteredLogs.length} events</span>
                    </div>

                    <div className="terminal-body">
                        {filteredLogs.length === 0 ? (
                            <div className="no-logs">
                                <Search size={32} color="var(--text-muted)" />
                                <p>No system events recorded yet.</p>
                                <span className="no-logs-hint">Logs will appear here automatically as you use the system.</span>
                            </div>
                        ) : (
                            filteredLogs.map((log) => (
                                <div
                                    key={log.id}
                                    className={`log-row-item ${log.level} clickable`}
                                    onClick={() => setSelectedLog(log)}
                                    title="Click to view details"
                                >
                                    <span className="log-cell-timestamp">[{log.timestamp}]</span>
                                    <span className={`log-cell-level badge-${log.level}`}>
                                        {getLevelIcon(log.level)}
                                        {log.level.toUpperCase()}
                                    </span>
                                    <span className="log-cell-message">{log.message}</span>
                                    <span className="log-cell-expand"><ExternalLink size={12} /></span>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </main>

            {/* Log Detail Modal */}
            {selectedLog && (
                <div className="log-detail-overlay" onClick={() => setSelectedLog(null)}>
                    <div className="log-detail-modal" onClick={(e) => e.stopPropagation()}>
                        <div className="log-detail-header">
                            <div className="log-detail-title-row">
                                <span className={`log-detail-level-badge badge-${selectedLog.level}`}>
                                    {getLevelIcon(selectedLog.level)}
                                    {selectedLog.level.toUpperCase()}
                                </span>
                            </div>
                            <button className="log-detail-close" onClick={() => setSelectedLog(null)}>
                                <X size={20} />
                            </button>
                        </div>

                        <div className="log-detail-body">
                            <h3 className="log-detail-message">{selectedLog.message}</h3>

                            <div className="log-detail-meta-grid">
                                <div className="log-detail-meta-item">
                                    <span className="meta-label">Timestamp</span>
                                    <span className="meta-value">{selectedLog.timestamp}</span>
                                </div>
                                <div className="log-detail-meta-item">
                                    <span className="meta-label">User</span>
                                    <span className="meta-value">@{selectedLog.user}</span>
                                </div>
                                <div className="log-detail-meta-item">
                                    <span className="meta-label">IP Address</span>
                                    <span className="meta-value">{selectedLog.ip}</span>
                                </div>
                                <div className="log-detail-meta-item">
                                    <span className="meta-label">Event ID</span>
                                    <span className="meta-value">EVT-{String(selectedLog.id).padStart(5, '0')}</span>
                                </div>
                            </div>

                            {selectedLog.details && (
                                <div className="log-detail-description">
                                    <span className="meta-label">Full Details</span>
                                    <p>{selectedLog.details}</p>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default LogViewer;
