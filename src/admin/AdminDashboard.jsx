import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Users, Video, ShieldAlert, FileClock, Terminal, ArrowRight, Activity, Cpu } from 'lucide-react';
import AdminHeader from './AdminHeader';
import './AdminDashboard.css';

const AdminDashboard = () => {
    const navigate = useNavigate();

    // Mock logs for the dashboard audit trail preview
    const [recentLogs, setRecentLogs] = useState([
        { id: 1, time: 'Just Now', category: 'AUTH', action: 'User login success', user: 'admin', level: 'info' },
        { id: 2, time: '10 mins ago', category: 'SURVEILLANCE', action: 'Camera 04 connection restored', user: 'system', level: 'info' },
        { id: 3, time: '30 mins ago', category: 'POLICY', action: 'Biometric threshold updated to 94%', user: 'admin', level: 'warning' },
        { id: 4, time: '1 hour ago', category: 'AUTH', action: 'New Investigator user registered', user: 'admin', level: 'info' }
    ]);

    // Live update effect for a dynamic command center feel
    useEffect(() => {
        const interval = setInterval(() => {
            const actions = [
                { category: 'SURVEILLANCE', action: 'Pings received from all 8 cameras', user: 'system', level: 'info' },
                { category: 'AUTH', action: 'Token expiration checked', user: 'system', level: 'info' },
                { category: 'DATABASE', action: 'Backup index refreshed successfully', user: 'system', level: 'info' }
            ];
            const randomAction = actions[Math.floor(Math.random() * actions.length)];
            setRecentLogs(prev => [
                { id: Date.now(), time: 'Just Now', ...randomAction },
                ...prev.slice(0, 3)
            ]);
        }, 15000);

        return () => clearInterval(interval);
    }, []);

    return (
        <div className="admin-dashboard-container">
            <AdminHeader />
            
            <main className="admin-dashboard-content">
                <section className="admin-welcome-section">
                    <div className="welcome-banner">
                        <div className="pulse-indicator"></div>
                        <h1>Command Center</h1>
                        <p>Argus System Administration & Operational Core</p>
                    </div>
                </section>

                <div className="admin-stats-grid">
                    <div className="admin-stat-card">
                        <Users size={24} className="stat-icon cyan" />
                        <span className="admin-stat-title">System Operators</span>
                        <span className="admin-stat-value">12</span>
                        <span className="admin-stat-sub">10 Investigators, 2 Admins</span>
                    </div>
                    <div className="admin-stat-card">
                        <Video size={24} className="stat-icon blue" />
                        <span className="admin-stat-title">Surveillance Feeds</span>
                        <span className="admin-stat-value">08</span>
                        <span className="admin-stat-sub">8 Online, 0 Offline</span>
                    </div>
                    <div className="admin-stat-card">
                        <Cpu size={24} className="stat-icon purple" />
                        <span className="admin-stat-title">Active Policies</span>
                        <span className="admin-stat-value">05</span>
                        <span className="admin-stat-sub">Security constraints active</span>
                    </div>
                    <div className="admin-stat-card">
                        <Activity size={24} className="stat-icon green" />
                        <span className="admin-stat-title">System Health</span>
                        <span className="admin-stat-value text-green">100%</span>
                        <span className="admin-stat-sub">All components operational</span>
                    </div>
                </div>

                <div className="admin-panels-layout">
                    <div className="quick-access-panel">
                        <h2>Administrative Controls</h2>
                        <p className="panel-desc">Quickly configure, review, or authorize system components.</p>
                        
                        <div className="control-cards-grid">
                            <div className="control-card" onClick={() => navigate('/admin/users')}>
                                <div className="card-header-icon">
                                    <Users size={22} color="var(--ice)" />
                                </div>
                                <div className="card-info">
                                    <h3>User Management</h3>
                                    <p>Add/edit system operator roles, permissions and suspend credentials.</p>
                                </div>
                                <ArrowRight className="card-arrow" size={18} />
                            </div>

                            <div className="control-card" onClick={() => navigate('/admin/surveillance')}>
                                <div className="card-header-icon">
                                    <Video size={22} color="var(--ice)" />
                                </div>
                                <div className="card-info">
                                    <h3>Surveillance Feeds</h3>
                                    <p>Access live security streams, map plotting, and feeds registration.</p>
                                </div>
                                <ArrowRight className="card-arrow" size={18} />
                            </div>

                            <div className="control-card" onClick={() => navigate('/admin/policies')}>
                                <div className="card-header-icon">
                                    <Cpu size={22} color="var(--ice)" />
                                </div>
                                <div className="card-info">
                                    <h3>Security Policies</h3>
                                    <p>Configure facial match thresholds, retention periods, and sessions.</p>
                                </div>
                                <ArrowRight className="card-arrow" size={18} />
                            </div>

                            <div className="control-card" onClick={() => navigate('/admin/logs')}>
                                <div className="card-header-icon">
                                    <FileClock size={22} color="var(--ice)" />
                                </div>
                                <div className="card-info">
                                    <h3>System Logs</h3>
                                    <p>Audit system operations, tracking history trails, and events stream.</p>
                                </div>
                                <ArrowRight className="card-arrow" size={18} />
                            </div>
                        </div>
                    </div>

                    <div className="audit-preview-panel">
                        <div className="panel-header-row">
                            <h2>Live Audit Stream</h2>
                            <button className="view-all-logs-btn" onClick={() => navigate('/admin/logs')}>
                                View All Logs
                            </button>
                        </div>
                        <p className="panel-desc">Real-time log events happening inside Argus system.</p>

                        <div className="mini-logs-list">
                            {recentLogs.map((log) => (
                                <div key={log.id} className={`mini-log-item ${log.level}`}>
                                    <div className="mini-log-meta">
                                        <Terminal size={14} className="terminal-log-icon" />
                                        <span className="log-category">[{log.category}]</span>
                                        <span className="log-time">{log.time}</span>
                                    </div>
                                    <div className="log-msg-row">
                                        <p className="log-msg">{log.action}</p>
                                        <span className="log-user">@{log.user}</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
};

export default AdminDashboard;
