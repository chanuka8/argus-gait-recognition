import React, { useState, useEffect } from 'react';
import { Search, Filter, Trash2, Download, AlertTriangle, Info, Skull, RefreshCcw } from 'lucide-react';
import AdminHeader from './AdminHeader';
import './LogViewer.css';

const LogViewer = () => {
    const defaultLogs = [
        { id: 1, timestamp: '2026-06-10 00:01:25', level: 'info', category: 'AUTH', message: 'Operator admin successfully authenticated via secure token', user: 'admin', ip: '192.168.1.1' },
        { id: 2, timestamp: '2026-06-09 23:58:44', level: 'info', category: 'SURVEILLANCE', message: 'Tracking frame sequence matched target metadata', user: 'system', ip: 'localhost' },
        { id: 3, timestamp: '2026-06-09 23:45:10', level: 'warning', category: 'DATABASE', message: 'Connection load spiked above 80% capacity during history query', user: 'system', ip: '127.0.0.1' },
        { id: 4, timestamp: '2026-06-09 23:30:15', level: 'info', category: 'POLICY', message: 'Facial identification match parameter set to 94%', user: 'admin', ip: '192.168.1.1' },
        { id: 5, timestamp: '2026-06-09 22:15:30', level: 'critical', category: 'AUTH', message: 'Failed login attempt detected from unknown host IP 203.0.113.43', user: 'unknown', ip: '203.0.113.43' },
        { id: 6, timestamp: '2026-06-09 21:04:12', level: 'info', category: 'SURVEILLANCE', message: 'Added CCTV feed 08 (Sector B West Corner)', user: 'admin', ip: '192.168.1.1' },
        { id: 7, timestamp: '2026-06-09 19:44:02', level: 'warning', category: 'SURVEILLANCE', message: 'Video drop frames detected on channel 03', user: 'system', ip: 'localhost' },
        { id: 8, timestamp: '2026-06-09 18:22:19', level: 'info', category: 'POLICY', message: 'Policy threshold for automatic cold-case archive changed to 365 days', user: 'admin', ip: '192.168.1.1' },
        { id: 9, timestamp: '2026-06-09 15:10:45', level: 'info', category: 'AUTH', message: 'Operator inv_john changed password successfully', user: 'inv_john', ip: '192.168.1.12' },
        { id: 10, timestamp: '2026-06-09 10:05:00', level: 'critical', category: 'DATABASE', message: 'Storage partition /dev/sda4 reached 92% capacity limit', user: 'system', ip: 'localhost' }
    ];

    const [logs, setLogs] = useState(() => {
        const localLogs = localStorage.getItem('argus_system_logs');
        return localLogs ? JSON.parse(localLogs) : defaultLogs;
    });

    const [searchTerm, setSearchTerm] = useState('');
    const [selectedLevel, setSelectedLevel] = useState('all');
    const [selectedCategory, setSelectedCategory] = useState('all');

    useEffect(() => {
        localStorage.setItem('argus_system_logs', JSON.stringify(logs));
    }, [logs]);

    const handleClearLogs = () => {
        if (window.confirm('Are you sure you want to purge all system logs? This action is irreversible.')) {
            setLogs([]);
        }
    };

    const handleResetLogs = () => {
        setLogs(defaultLogs);
    };

    const handleExportLogs = () => {
        const headers = 'ID,Timestamp,Level,Category,Message,User,IP Address\n';
        const csvContent = logs.map(log => 
            `"${log.id}","${log.timestamp}","${log.level}","${log.category}","${log.message.replace(/"/g, '""')}","${log.user}","${log.ip}"`
        ).join('\n');
        
        const blob = new Blob([headers + csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.setAttribute('href', url);
        link.setAttribute('download', `argus_audit_logs_${Date.now()}.csv`);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
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
                              log.user.toLowerCase().includes(searchTerm.toLowerCase()) ||
                              log.category.toLowerCase().includes(searchTerm.toLowerCase());
        const matchesLevel = selectedLevel === 'all' || log.level === selectedLevel;
        const matchesCategory = selectedCategory === 'all' || log.category === selectedCategory;

        return matchesSearch && matchesLevel && matchesCategory;
    });

    return (
        <div className="log-viewer-page">
            <AdminHeader />

            <main className="log-viewer-content">
                <div className="log-header-row">
                    <div className="title-group">
                        <h1>System Audit Logs</h1>
                        <p>Track all platform authentication, surveillance connections, policy modifications and security breaches.</p>
                    </div>
                    <div className="log-actions">
                        <button className="log-action-btn reset" onClick={handleResetLogs} title="Reset to Defaults">
                            <RefreshCcw size={16} />
                            <span>Reset Feed</span>
                        </button>
                        <button className="log-action-btn export" onClick={handleExportLogs} title="Export to CSV">
                            <Download size={16} />
                            <span>Export CSV</span>
                        </button>
                        <button className="log-action-btn purge" onClick={handleClearLogs} title="Clear Logs">
                            <Trash2 size={16} />
                            <span>Purge Logs</span>
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

                        <div className="filter-select-group">
                            <Filter size={14} color="var(--text-muted)" />
                            <label>Category:</label>
                            <select value={selectedCategory} onChange={(e) => setSelectedCategory(e.target.value)}>
                                <option value="all">ALL CATEGORIES</option>
                                <option value="AUTH">AUTH</option>
                                <option value="SURVEILLANCE">SURVEILLANCE</option>
                                <option value="POLICY">POLICY</option>
                                <option value="DATABASE">DATABASE</option>
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
                                <p>No audit events match current criteria.</p>
                            </div>
                        ) : (
                            filteredLogs.map((log) => (
                                <div key={log.id} className={`log-row-item ${log.level}`}>
                                    <span className="log-cell-timestamp">[{log.timestamp}]</span>
                                    <span className={`log-cell-level badge-${log.level}`}>
                                        {getLevelIcon(log.level)}
                                        {log.level.toUpperCase()}
                                    </span>
                                    <span className="log-cell-category">[{log.category}]</span>
                                    <span className="log-cell-message">{log.message}</span>
                                    <span className="log-cell-user">@{log.user}</span>
                                    <span className="log-cell-ip">({log.ip})</span>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </main>
        </div>
    );
};

export default LogViewer;
