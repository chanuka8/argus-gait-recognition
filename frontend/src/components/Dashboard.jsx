import React, { useState, useEffect } from 'react';
import {
    User, Bell, Search, Video, Clock, Activity,
    ShieldAlert, Eye, MapPin, ChevronRight, Radio
} from 'lucide-react';
import logo from '../assets/logo.png';
import './Dashboard.css';
import { useNavigate } from 'react-router-dom';
import MapComponent from './Map';
import Notifications from './Notifications';
import UserProfileModal from './UserProfileModal';
import { useAuth } from '../contexts/AuthContext';
import { db } from '../firebaseConfig';
import { collection, getDocs, onSnapshot, query } from 'firebase/firestore';
import GaitSystemStatus from './GaitSystemStatus';
import RecognitionEvents from './RecognitionEvents';

const CountUp = ({ end, duration }) => {
    const [count, setCount] = useState(0);

    useEffect(() => {
        let startTime = null;
        let animationFrame;
        const finalDuration = duration || Math.min(2000, Math.max(800, end * 250));

        const animate = (currentTime) => {
            if (!startTime) startTime = currentTime;
            const progress = currentTime - startTime;
            const percentage = Math.min(progress / finalDuration, 1);
            const easeOutCubic = 1 - Math.pow(1 - percentage, 3);
            const currentCount = Math.floor(end * easeOutCubic);

            setCount(currentCount);

            if (progress < finalDuration) {
                animationFrame = requestAnimationFrame(animate);
            } else {
                setCount(end);
            }
        };

        animationFrame = requestAnimationFrame(animate);
        return () => cancelAnimationFrame(animationFrame);
    }, [end, duration]);

    return (
        <span>{count.toString().padStart(2, '0')}</span>
    );
};

const Dashboard = () => {
    const navigate = useNavigate();
    const { currentUser } = useAuth();
    const [showNotifications, setShowNotifications] = useState(false);
    const [showProfile, setShowProfile] = useState(false);
    const [cases, setCases] = useState([]);
    const [detections, setDetections] = useState([]);

    useEffect(() => {
        const fetchCases = async () => {
            try {
                const snapshot = await getDocs(collection(db, 'victims'));
                const casesList = [];
                snapshot.forEach((doc) => {
                    const data = doc.data();
                    casesList.push({
                        id: doc.id,
                        ...data,
                        status: data.status || 'Investigating',
                        caseType: data.caseType || ''
                    });
                });
                setCases(casesList);
            } catch (error) {
                console.error('Error fetching cases stats:', error);
            }
        };

        fetchCases();

        const qDet = query(collection(db, 'detections'));
        const unsubDet = onSnapshot(qDet, (snapshot) => {
            const detList = [];
            snapshot.forEach((doc) => {
                detList.push({ id: doc.id, ...doc.data() });
            });
            detList.reverse();
            setDetections(detList.slice(0, 6));
        }, (err) => {
            console.error('Error subscribing to live detections:', err);
        });

        return () => unsubDet();
    }, []);

    const totalCases = cases.length;
    const missingCases = cases.filter(c => c.caseType?.toLowerCase() === 'missing').length;
    const investigatingCases = cases.filter(c => c.status?.toLowerCase() === 'investigating').length;
    const foundCases = cases.filter(c => c.status?.toLowerCase() === 'found' || c.status?.toLowerCase() === 'closed').length;
    const coldCases = cases.filter(c => c.status?.toLowerCase() === 'cold').length;

    return (
        <div className="dashboard-container">
            <Notifications isOpen={showNotifications} onClose={() => setShowNotifications(false)} />
            <UserProfileModal isOpen={showProfile} onClose={() => setShowProfile(false)} />

            <header className="command-header">
                <div className="header-brand-group">
                    <img src={logo} alt="ARGUS Logo" className="header-logo" />
                    <div className="brand-titles">
                        <span className="system-code">ARGUS-V0.1 // COMMAND</span>
                        <h1 className="header-title">MISSING PERSONS RECON SYSTEM</h1>
                    </div>
                </div>

                <div className="header-controls-group">
                    <div className="user-profile-widget" onClick={() => setShowProfile(true)}>
                        <div className="avatar-circle">
                            <User size={16} />
                        </div>
                        <div className="user-text-info">
                            <span className="user-name">{currentUser?.displayName || currentUser?.email || 'Officer'}</span>
                            <span className="user-role">INVESTIGATOR</span>
                        </div>
                    </div>

                    <div className="notification-wrapper">
                        <button
                            className="icon-btn notification-btn"
                            onClick={() => setShowNotifications(!showNotifications)}
                            title="Notifications"
                        >
                            <Bell size={18} />
                            {detections.length > 0 && <span className="notification-badge">{detections.length}</span>}
                        </button>
                    </div>
                </div>
            </header>

            <main className="command-workspace">
                {/* LEFT PANE: TACTICAL MAP & HUD TELEMETRY */}
                <section className="tactical-map-pane">
                    <div className="map-hud-ribbon">
                        <div className="hud-metric-pill total">
                            <span className="hud-label">TOTAL CASES</span>
                            <span className="hud-value white"><CountUp end={totalCases} /></span>
                        </div>
                        <div className="hud-metric-pill missing">
                            <span className="hud-label">MISSING</span>
                            <span className="hud-value red"><CountUp end={missingCases} /></span>
                        </div>
                        <div className="hud-metric-pill active-case">
                            <span className="hud-label">INVESTIGATING</span>
                            <span className="hud-value yellow"><CountUp end={investigatingCases} /></span>
                        </div>
                        <div className="hud-metric-pill found">
                            <span className="hud-label">RESOLVED / FOUND</span>
                            <span className="hud-value green"><CountUp end={foundCases} /></span>
                        </div>
                        <div className="hud-metric-pill cold">
                            <span className="hud-label">COLD CASES</span>
                            <span className="hud-value blue"><CountUp end={coldCases} /></span>
                        </div>
                    </div>

                    <div className="tactical-map-viewport">
                        <MapComponent cases={cases} />
                        <div className="zone-status-bar">
                            <div className="status-indicator">
                                <Radio className="pulsing-radio" size={16} />
                                <span>CCTV SURVEILLANCE GRID ACTIVE</span>
                            </div>
                            <span className="zone-ready-text">ARGUS Gait Biometric AI Integrated</span>
                        </div>
                    </div>
                </section>

                {/* RIGHT PANE: OPERATIONS DOCK & LIVE FEED */}
                <aside className="operations-dock">
                    <GaitSystemStatus />

                    <div className="quick-command-section">
                        <h3 className="dock-section-title">OPERATIONAL COMMANDS</h3>
                        <div className="command-cards-grid">
                            <div className="command-action-card primary" onClick={() => navigate('/report-case')}>
                                <div className="card-icon-wrapper red-glow">
                                    <Search size={22} color="#FF5252" />
                                </div>
                                <div className="card-text-wrapper">
                                    <h4>Find a Missing Person</h4>
                                    <p>Deploy active search profile & intelligence</p>
                                </div>
                                <ChevronRight size={18} className="chevron" />
                            </div>

                            <div className="command-action-card secondary" onClick={() => navigate('/cctv-network')}>
                                <div className="card-icon-wrapper cyan-glow">
                                    <Video size={22} color="#00E5FF" />
                                </div>
                                <div className="card-text-wrapper">
                                    <h4>CCTV Zones</h4>
                                    <p>Configure AI sentinel nodes & surveillance perimeters</p>
                                </div>
                                <ChevronRight size={18} className="chevron" />
                            </div>

                            <div className="command-action-card tertiary" onClick={() => navigate('/history')}>
                                <div className="card-icon-wrapper ice-glow">
                                    <Clock size={22} color="#42A5F5" />
                                </div>
                                <div className="card-text-wrapper">
                                    <h4>Investigation History</h4>
                                    <p>Access archived case logs & sighting trails</p>
                                </div>
                                <ChevronRight size={18} className="chevron" />
                            </div>
                        </div>
                    </div>

                    <div className="live-telemetry-section">
                        <div className="feed-header">
                            <div className="header-left-group">
                                <Activity size={18} color="#00E5FF" className="activity-spin" />
                                <h3>LIVE RECON & AI ALERTS</h3>
                            </div>
                            <span className="live-status-badge">● LIVE STREAM</span>
                        </div>
                        <div className="activity-feed-list">
                            {detections.length > 0 ? (
                                detections.map((det, index) => (
                                    <div key={det.id || index} className="feed-alert-item" onClick={() => det.caseId && navigate(`/case/${det.caseId}`)}>
                                        <div className="alert-icon-box">
                                            <ShieldAlert size={18} color="#FF6F00" />
                                        </div>
                                        <div className="alert-details">
                                            <div className="alert-title-row">
                                                <strong>{det.victimName || 'Target Detected'}</strong>
                                                <span className="conf-tag">{det.confidenceScore ? `${Math.round(det.confidenceScore * 100)}% Match` : 'AI Lock'}</span>
                                            </div>
                                            <div className="alert-sub-row">
                                                <span><Eye size={12} /> {det.cameraId || 'CCTV Node'}</span>
                                                <span><MapPin size={12} /> {det.locationName || 'GPS Locked'}</span>
                                            </div>
                                        </div>
                                    </div>
                                ))
                            ) : (
                                <div className="standby-telemetry-box">
                                    <div className="telemetry-item">
                                        <span className="node-badge">Z01 COLOMBO</span>
                                        <span>Terminal Intersections & Harbor — <strong>SECURE</strong></span>
                                    </div>
                                    <div className="telemetry-item">
                                        <span className="node-badge">Z02 GALLE</span>
                                        <span>Southern Expressway Checkpoints — <strong>ONLINE</strong></span>
                                    </div>
                                    <div className="telemetry-item">
                                        <span className="node-badge">Z03 KANDY</span>
                                        <span>Highland Municipal Borders — <strong>ONLINE</strong></span>
                                    </div>
                                    <p className="standby-note">No immediate AI target alarms in progress. Real-time recognition algorithm active.</p>
                                </div>
                            )}
                        </div>
                    </div>

                    <RecognitionEvents />
                </aside>
            </main>
        </div>
    );
};

export default Dashboard;
