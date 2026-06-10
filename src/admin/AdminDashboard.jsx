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

    // Fetch operators from Firebase
    const [adminCount, setAdminCount] = useState(0);
    const [investigatorCount, setInvestigatorCount] = useState(0);
    const [operatorsLoading, setOperatorsLoading] = useState(true);

    // Fetch surveillance feeds from Firebase
    const [cameraCount, setCameraCount] = useState(0);
    const [onlineCameraCount, setOnlineCameraCount] = useState(0);
    const [camerasLoading, setCamerasLoading] = useState(true);

    useEffect(() => {
        const fetchCameras = async () => {
            try {
                const snapshot = await getDocs(collection(db, 'cameras'));
                let total = snapshot.size;
                let online = 0;
                snapshot.forEach((doc) => {
                    const data = doc.data();
                    const status = data.status || '';
                    if (status.toLowerCase() === 'online') {
                        online++;
                    }
                });
                setCameraCount(total);
                setOnlineCameraCount(online);
            } catch (error) {
                console.error('Error fetching cameras count:', error);
            } finally {
                setCamerasLoading(false);
            }
        };
        fetchCameras();
    }, []);

    useEffect(() => {
        const fetchOperatorsCount = async () => {
            try {
                const adminSnapshot = await getDocs(collection(db, 'admins'));
                const invSnapshot = await getDocs(collection(db, 'investigators'));
                setAdminCount(adminSnapshot.size);
                setInvestigatorCount(invSnapshot.size);
            } catch (error) {
                console.error('Error fetching operators count:', error);
            } finally {
                setOperatorsLoading(false);
            }
        };
        fetchOperatorsCount();
    }, []);

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
                        status: data.status || 'Investigating',
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

    const activeCases = cases.filter(c => c.status?.toLowerCase() === 'investigating' || c.status?.toLowerCase() === 'cold');

    return (
        <div className="admin-dashboard-container">
            <AdminHeader />

            <main className="admin-dashboard-content">
                <section className="admin-welcome-section">
                    <div className="welcome-banner">
                        <div className="pulse-indicator"></div>
                        <h1>Control Center</h1>
                    </div>
                </section>

                <div className="admin-stats-grid">
                    <div className="admin-stat-card">
                        <Users size={24} className="stat-icon cyan" />
                        <span className="admin-stat-title">System Operators</span>
                        <span className="admin-stat-value">
                            {operatorsLoading ? '...' : String(adminCount + investigatorCount).padStart(2, '0')}
                        </span>
                        <span className="admin-stat-sub">
                            {operatorsLoading ? 'Loading counts...' : `${investigatorCount} Investigators, ${adminCount} Admins`}
                        </span>
                    </div>
                    <div className="admin-stat-card">
                        <Video size={24} className="stat-icon blue" />
                        <span className="admin-stat-title">Surveillance Feeds</span>
                        <span className="admin-stat-value">
                            {camerasLoading ? '...' : String(cameraCount).padStart(2, '0')}
                        </span>
                        <span className="admin-stat-sub">
                            {camerasLoading ? 'Loading feeds...' : `${onlineCameraCount} Online, ${cameraCount - onlineCameraCount} Offline`}
                        </span>
                    </div>
                    <div className="admin-stat-card">
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

        </div>
    );
};

export default AdminDashboard;
