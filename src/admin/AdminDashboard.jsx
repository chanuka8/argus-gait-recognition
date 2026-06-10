import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { Users, Video, FileClock, Terminal, ArrowRight, Activity, Briefcase, X } from 'lucide-react';
import AdminHeader from './AdminHeader';
import { getLogs } from '../utils/logService';
import { db } from '../firebaseConfig';
import { collection, getDocs } from 'firebase/firestore';
import './AdminDashboard.css';

// Helper to convert log timestamp to relative time
const formatTimeAgo = (timestamp) => {
    const now = new Date();
    const logTime = new Date(timestamp.replace(' ', 'T'));
    const diffMs = now - logTime;
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);
    const diffHr = Math.floor(diffMin / 60);

    if (diffMin < 1) return 'Just Now';
    if (diffMin < 60) return `${diffMin} min ago`;
    if (diffHr < 24) return `${diffHr} hr ago`;
    return timestamp.slice(0, 10);
};

const AdminDashboard = () => {
    const navigate = useNavigate();

    // Pull recent logs from centralized log service
    const [recentLogs, setRecentLogs] = useState([]);

    const refreshDashboardLogs = useCallback(() => {
        const allLogs = getLogs();
        const latest = allLogs.slice(0, 4).map(log => ({
            id: log.id,
            time: formatTimeAgo(log.timestamp),
            category: 'SYSTEM',
            action: log.message,
            user: log.user,
            level: log.level
        }));
        setRecentLogs(latest);
    }, []);

    useEffect(() => {
        refreshDashboardLogs();

        const handleLogUpdate = () => refreshDashboardLogs();
        window.addEventListener('argus-log-update', handleLogUpdate);

        return () => window.removeEventListener('argus-log-update', handleLogUpdate);
    }, [refreshDashboardLogs]);

    // Fetch cases from Firebase
    const [cases, setCases] = useState([]);
    const [casesLoading, setCasesLoading] = useState(true);
    const [showCasesModal, setShowCasesModal] = useState(false);

    useEffect(() => {
        const fetchCases = async () => {
            try {
                const snapshot = await getDocs(collection(db, 'victims'));
                const casesList = [];
                snapshot.forEach((doc) => {
                    const data = doc.data();
                    casesList.push({
                        id: doc.id,
                        caseId: data.caseId || doc.id,
                        caseType: data.caseType || 'N/A',
                        name: data.name || 'Unknown',
                        nic: data.nic || 'N/A',
                        age: data.age || 'N/A',
                        gender: data.gender || 'N/A',
                        status: data.status || 'Active',
                        createdAt: data.createdAt?.toDate ? data.createdAt.toDate().toLocaleString() : 'N/A'
                    });
                });
                setCases(casesList);
            } catch (error) {
                console.error('Error fetching cases:', error);
            } finally {
                setCasesLoading(false);
            }
        };

        fetchCases();
    }, []);

    const activeCases = cases.filter(c => c.status === 'Active');

    const getStatusColor = (status) => {
        switch (status?.toLowerCase()) {
            case 'active': return 'status-active';
            case 'missing': return 'status-missing';
            case 'investigating': return 'status-investigating';
            case 'found': return 'status-found';
            case 'closed': return 'status-closed';
            default: return 'status-default';
        }
    };

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
                    <div className="admin-stat-card clickable-card" onClick={() => setShowCasesModal(true)}>
                        <Briefcase size={24} className="stat-icon purple" />
                        <span className="admin-stat-title">Active Cases</span>
                        <span className="admin-stat-value">
                            {casesLoading ? '...' : String(activeCases.length).padStart(2, '0')}
                        </span>
                        <span className="admin-stat-sub">
                            {casesLoading ? 'Loading...' : `${cases.length} total cases in system`}
                        </span>
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
                                    <Briefcase size={22} color="var(--ice)" />
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
                            {recentLogs.length === 0 ? (
                                <div className="mini-logs-empty">
                                    <p>No system events recorded yet.</p>
                                </div>
                            ) : (
                                recentLogs.map((log) => (
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
                                ))
                            )}
                        </div>
                    </div>
                </div>
            </main>

            {/* Cases List Modal */}
            {showCasesModal && (
                <div className="cases-modal-overlay" onClick={() => setShowCasesModal(false)}>
                    <div className="cases-modal" onClick={(e) => e.stopPropagation()}>
                        <div className="cases-modal-header">
                            <div className="modal-title-flex">
                                <Briefcase size={22} className="modal-title-icon" />
                                <h2>Active Cases</h2>
                            </div>
                            <button className="cases-modal-close" onClick={() => setShowCasesModal(false)}>
                                <X size={20} />
                            </button>
                        </div>

                        <div className="cases-modal-body">
                            {casesLoading ? (
                                <div className="cases-loading">
                                    <div className="loading-spinner"></div>
                                    <p>Retrieving operational case logs...</p>
                                </div>
                            ) : cases.length === 0 ? (
                                <div className="cases-empty">
                                    <Briefcase size={36} className="empty-icon" />
                                    <p>No case logs currently active in system.</p>
                                </div>
                            ) : (
                                <div className="cases-table-wrapper">
                                    <table className="cases-table">
                                        <thead>
                                            <tr>
                                                <th>Case ID</th>
                                                <th>Case Type</th>
                                                <th>Status</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {cases.map((c) => (
                                                <tr key={c.id}>
                                                    <td className="case-id-cell">{c.caseId}</td>
                                                    <td className="case-type-cell">{c.caseType}</td>
                                                    <td>
                                                        <span className={`case-status-badge ${getStatusColor(c.status)}`}>
                                                            <span className="status-dot"></span>
                                                            {c.status}
                                                        </span>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default AdminDashboard;
