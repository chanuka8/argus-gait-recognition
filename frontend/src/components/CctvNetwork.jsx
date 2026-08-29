import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    ArrowLeft, Video, Radio, Plus, MapPin, Wifi,
    Cpu, Eye, X, AlertTriangle, CheckCircle, Bell, User as UserIcon, Play, Square, Loader, RefreshCw
} from 'lucide-react';
import { SURVEILLANCE_ZONES, CCTV_CAMERAS as INITIAL_CAMERAS, triggerSimulatedDetection } from '../utils/cctvService';
import { getDeviceGPS } from '../utils/geoService';
import { db } from '../firebaseConfig';
import { collection, getDocs } from 'firebase/firestore';
import logo from '../assets/logo.png';
import Notifications from './Notifications';
import UserProfileModal from './UserProfileModal';
import { useAuth } from '../hooks/useAuth';
import { useGait } from '../hooks/useGait';
import { getStreamUrl } from '../config/apiConfig';
import './CctvNetwork.css';
import './History.css';

const LiveCameraFeed = ({ cameraId, cameraName, workerActive, telemetry }) => {
    const [streamError, setStreamError] = useState(false);
    const [isLoaded, setIsLoaded] = useState(false);
    const [retryKey, setRetryKey] = useState(0);

    const handleRetry = () => {
        setStreamError(false);
        setIsLoaded(false);
        setRetryKey(k => k + 1);
    };

    if (!workerActive) {
        return (
            <div className="simulated-feed-box">
                <div className="feed-watermark">
                    <span>REC ⬤ [STANDBY]</span>
                    <span>{new Date().toLocaleTimeString()}</span>
                </div>
                <div className="radar-target-box" style={{ borderColor: '#00E5FF', color: '#00E5FF' }}>
                    <span>⚡ STANDBY (CLICK 'START WORKER' TO STREAM)</span>
                </div>
                <div className="feed-telemetry">
                    Hardware Node: {cameraId} | Status: STANDBY
                </div>
            </div>
        );
    }

    const streamUrl = `${getStreamUrl(cameraId)}?t=${retryKey}`;

    return (
        <div className="simulated-feed-box" style={{ position: 'relative', overflow: 'hidden', minHeight: '180px' }}>
            <div className="feed-watermark" style={{ zIndex: 3 }}>
                <span style={{ color: '#06D6A0', fontWeight: 800 }}>⬤ LIVE STREAM [ACTIVE]</span>
                <span>{new Date().toLocaleTimeString()}</span>
            </div>

            {!streamError && (
                <img
                    key={retryKey}
                    src={streamUrl}
                    alt={`Live Stream - ${cameraName}`}
                    style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        objectFit: 'cover',
                        zIndex: 1,
                        opacity: isLoaded ? 1 : 0,
                        transition: 'opacity 0.25s ease-in-out',
                    }}
                    onLoad={() => {
                        setIsLoaded(true);
                        setStreamError(false);
                    }}
                    onError={() => {
                        setStreamError(true);
                        setIsLoaded(false);
                    }}
                />
            )}

            {(!isLoaded || streamError) && (
                <div
                    style={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '8px',
                        background: 'rgba(7, 18, 30, 0.92)',
                        zIndex: 2,
                        padding: '1rem',
                        textAlign: 'center',
                    }}
                >
                    {!streamError ? (
                        <>
                            <Loader size={24} className="activity-spin" color="#00E5FF" />
                            <span style={{ color: '#00E5FF', fontSize: '0.85rem', fontWeight: 600 }}>
                                Initializing MJPEG Stream...
                            </span>
                        </>
                    ) : (
                        <>
                            <AlertTriangle size={24} color="#FFD166" />
                            <span style={{ color: '#FFD166', fontSize: '0.82rem', fontWeight: 600 }}>
                                Stream Disconnected or Initializing
                            </span>
                            <button
                                type="button"
                                onClick={handleRetry}
                                style={{
                                    background: 'rgba(0, 229, 255, 0.15)',
                                    border: '1px solid #00E5FF',
                                    color: '#00E5FF',
                                    padding: '4px 10px',
                                    borderRadius: '4px',
                                    fontSize: '0.75rem',
                                    cursor: 'pointer',
                                    display: 'flex',
                                    alignItems: 'center',
                                    gap: '4px',
                                }}
                            >
                                <RefreshCw size={12} /> Retry Stream
                            </button>
                        </>
                    )}
                </div>
            )}

            <div className="feed-telemetry" style={{ zIndex: 3, background: 'rgba(5, 15, 25, 0.8)' }}>
                FPS: {telemetry?.fps || 15} | Tracks: {telemetry?.active_tracks ?? 0} | Clients: {telemetry?.active_clients ?? 1} | ByGaitLight 256D
            </div>
        </div>
    );
};

const CctvNetwork = ({ isAdmin = false }) => {
    const navigate = useNavigate();
    const { currentUser } = useAuth();
    const { events, cameras: activeGaitCameras, startCamera, stopCamera } = useGait();

    const [showNotifications, setShowNotifications] = useState(false);
    const [showProfile, setShowProfile] = useState(false);
    const [activeZoneId, setActiveZoneId] = useState(SURVEILLANCE_ZONES[0]?.id || 'Z01');
    const [allCameras, setAllCameras] = useState([]);
    const [casesList, setCasesList] = useState([]);
    const [showAddModal, setShowAddModal] = useState(false);
    const [selectedCamStream, setSelectedCamStream] = useState(null);
    const [isDetectingGPS, setIsDetectingGPS] = useState(false);
    const [actionLoading, setActionLoading] = useState({});
    const [actionErrors, setActionErrors] = useState({});
    const [actionFeedback, setActionFeedback] = useState(null);

    const [newCam, setNewCam] = useState({
        id: `CCTV-${Math.floor(100 + Math.random() * 900)}`,
        name: '',
        lat: '',
        lng: '',
        resolution: '4K IR PTZ Stream & AI Face Recognition',
        ip: `192.168.${Math.floor(10 + Math.random() * 80)}.${Math.floor(10 + Math.random() * 240)}`,
        status: 'ONLINE'
    });

    useEffect(() => {
        const storedCustom = localStorage.getItem('argus_custom_cctv_nodes');
        let parsedCustom = [];
        if (storedCustom) {
            try {
                parsedCustom = JSON.parse(storedCustom);
            } catch (e) {
                console.error("Error parsing custom CCTV nodes:", e);
            }
        }
        setAllCameras([...INITIAL_CAMERAS, ...parsedCustom]);
    }, []);

    useEffect(() => {
        const fetchCases = async () => {
            try {
                const snapshot = await getDocs(collection(db, 'victims'));
                const list = [];
                snapshot.forEach(doc => {
                    list.push({ id: doc.id, ...doc.data() });
                });
                setCasesList(list);
            } catch (error) {
                console.error("Error loading target cases:", error);
            }
        };
        fetchCases();
    }, []);

    const activeZone = SURVEILLANCE_ZONES.find(z => z.id === activeZoneId) || SURVEILLANCE_ZONES[0];
    const activeZoneCameras = allCameras.filter(cam => cam.zoneId === activeZoneId || (!cam.zoneId && activeZoneId === 'Z01'));

    const handleSimulateAlert = async (cam) => {
        try {
            const targetSubject = casesList.length > 0
                ? casesList[Math.floor(Math.random() * casesList.length)]
                : { id: 'DEMO-TARGET', name: 'Simulated Target Subject (Demo)' };

            await triggerSimulatedDetection(cam, targetSubject);
            setActionFeedback({
                type: 'success',
                message: `🚨 AI SURVEILLANCE ALARM TRIGGERED!\nNode [${cam.id}] ${cam.name} matched target ${targetSubject.name || 'Subject'}`
            });
        } catch (err) {
            setActionFeedback({ type: 'error', message: "Simulation Failed: " + err.message });
        }
    };

    const getGaitCamera = (camId) => {
        return activeGaitCameras.find(c => c.camera_id === camId);
    };

    const isGaitWorkerActive = (camId) => {
        const cam = getGaitCamera(camId);
        return Boolean(cam && (cam.status === 'ACTIVE' || cam.status === 'connected'));
    };

    const handleToggleGaitWorker = async (cam) => {
        setActionLoading(prev => ({ ...prev, [cam.id]: true }));
        setActionErrors(prev => ({ ...prev, [cam.id]: null }));
        setActionFeedback(null);
        try {
            if (isGaitWorkerActive(cam.id)) {
                await stopCamera(cam.id);
                setActionFeedback({ type: 'success', message: `Camera worker [${cam.id}] stopped.` });
            } else {
                const source = (cam.source && String(cam.source).trim() !== '')
                    ? String(cam.source).trim()
                    : 'auto';
                const res = await startCamera(cam.id, source, cam.name, cam.zoneId || activeZoneId);
                const detectedType = (res?.source_type === 'webcam' || res?.resolved_source_type === 'webcam' || res?.resolved_source_type === 'usb') ? 'Webcam' : 'RTSP';
                setActionFeedback({
                    type: 'success',
                    message: `Camera [${cam.id}] connected (Auto-detected Source: ${detectedType})`
                });
            }
        } catch (err) {
            console.error(`Camera action failed for ${cam.id}:`, err);
            const userMsg = err.message || 'Unable to connect to camera source';
            setActionErrors(prev => ({ ...prev, [cam.id]: userMsg }));
            setActionFeedback({ type: 'error', message: `Camera Connection Failed: ${userMsg}` });
        } finally {
            setActionLoading(prev => ({ ...prev, [cam.id]: false }));
        }
    };

    const handleAutoDetectGPS = async () => {
        setIsDetectingGPS(true);
        try {
            const pos = await getDeviceGPS();
            setNewCam(prev => ({
                ...prev,
                lat: pos.lat.toFixed(6),
                lng: pos.lng.toFixed(6)
            }));
        } catch (err) {
            alert("Unable to acquire GPS coordinates: " + err.message);
        } finally {
            setIsDetectingGPS(false);
        }
    };

    const handleAddCameraSubmit = (e) => {
        e.preventDefault();
        if (!newCam.name || !newCam.lat || !newCam.lng) {
            alert("Please fill in camera landmark name and GPS coordinates.");
            return;
        }

        const newCamObj = {
            ...newCam,
            lat: parseFloat(newCam.lat),
            lng: parseFloat(newCam.lng),
            zoneId: activeZoneId,
            isCustom: true
        };

        const updatedAll = [...allCameras, newCamObj];
        setAllCameras(updatedAll);

        const customOnly = updatedAll.filter(c => !INITIAL_CAMERAS.some(ic => ic.id === c.id));
        localStorage.setItem('argus_custom_cctv_nodes', JSON.stringify(customOnly));

        setShowAddModal(false);
        setNewCam({
            id: `CCTV-${Math.floor(100 + Math.random() * 900)}`,
            name: '',
            lat: '',
            lng: '',
            resolution: '4K IR PTZ Stream & AI Face Recognition',
            ip: `192.168.${Math.floor(10 + Math.random() * 80)}.${Math.floor(10 + Math.random() * 240)}`,
            status: 'ONLINE'
        });
        setActionFeedback({ type: 'success', message: `Node ${newCamObj.id} deployed to ${activeZone.name}. Source will be auto-detected on connect.` });
    };

    const streamEvents = selectedCamStream
        ? events.filter(e => e.camera_id === selectedCamStream.id || e.camera_id === 'upload-image')
        : [];

    return (
        <div className="cctv-network-container">
            <Notifications isOpen={showNotifications} onClose={() => setShowNotifications(false)} />
            <UserProfileModal isOpen={showProfile} onClose={() => setShowProfile(false)} />

            <header className="history-header">
                <div className="history-header-left">
                    <button className="history-back-btn" onClick={() => navigate(isAdmin ? '/admin/dashboard' : '/dashboard')}>
                        <ArrowLeft size={24} />
                    </button>
                    <img src={logo} alt="Argus Logo" className="history-logo" />
                    <span className="history-title-text">ARGUS</span>
                </div>
                <div className="history-header-right">
                    <div className="user-profile" onClick={() => setShowProfile(true)} style={{ cursor: 'pointer' }}>
                        <UserIcon size={22} fill="#d6e4ea" color="#d6e4ea" />
                        <span>{currentUser?.displayName || currentUser?.email || 'Operator'}</span>
                    </div>
                    <Bell
                        size={22}
                        className="notification-bell"
                        fill="#5ce1e6"
                        color="#5ce1e6"
                        onClick={() => setShowNotifications(true)}
                        style={{ cursor: 'pointer' }}
                    />
                </div>
            </header>

            <main className="cctv-body">
                <div className="cctv-nav-header">
                    <div>
                        <h1 style={{ margin: 0, fontSize: '1.8rem', fontWeight: 800, display: 'flex', alignItems: 'center', gap: '10px' }}>
                            <Video size={28} color="#00E5FF" />
                            <span>CCTV Zones & Live Camera Stream</span>
                        </h1>
                        <p style={{ margin: '4px 0 0 0', fontSize: '0.9rem', color: 'rgba(255,255,255,0.6)' }}>
                            Real-time ByGaitLight & UNet biometric surveillance network with automatic camera-source detection
                        </p>
                    </div>
                    <div className="nav-stats-pills" style={{ display: 'flex', gap: '12px', flexWrap: 'wrap' }}>
                        <div className="status-pill active">
                            <span className="status-dot"></span>
                            <span>Gait AI Engine [ONLINE]</span>
                        </div>
                        <div className="status-pill">
                            <Wifi size={14} color="#00E5FF" />
                            <span>Configured Nodes: {allCameras.length}</span>
                        </div>
                        <div className="status-pill">
                            <Radio size={14} color="#5CE1E6" />
                            <span>Active Camera Workers: {activeGaitCameras.length}</span>
                        </div>
                    </div>
                </div>

                {actionFeedback && (
                    <div
                        style={{
                            margin: '0 0 1rem 0',
                            padding: '0.85rem 1.25rem',
                            borderRadius: '8px',
                            fontSize: '0.9rem',
                            fontWeight: 600,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            background: actionFeedback.type === 'success' ? 'rgba(6, 214, 160, 0.15)' : 'rgba(255, 107, 107, 0.15)',
                            border: actionFeedback.type === 'success' ? '1px solid #06D6A0' : '1px solid #FF6B6B',
                            color: actionFeedback.type === 'success' ? '#06D6A0' : '#FF6B6B',
                        }}
                    >
                        <span>{actionFeedback.message}</span>
                        <button onClick={() => setActionFeedback(null)} style={{ background: 'transparent', border: 'none', color: 'inherit', cursor: 'pointer' }}>
                            <X size={16} />
                        </button>
                    </div>
                )}

                <div className="zone-tabs-section">
                    <div className="tabs-list">
                        {SURVEILLANCE_ZONES.map((z) => {
                            const isSelected = z.id === activeZoneId;
                            const count = allCameras.filter(c => c.zoneId === z.id || (!c.zoneId && z.id === 'Z01')).length;
                            return (
                                <button
                                    key={z.id}
                                    type="button"
                                    className={`zone-tab-btn ${isSelected ? 'selected' : ''}`}
                                    onClick={() => setActiveZoneId(z.id)}
                                >
                                    <Radio size={16} color={isSelected ? '#00E5FF' : '#888'} />
                                    <span>{z.id}: {z.name.split('(')[0].trim()} ({count})</span>
                                </button>
                            );
                        })}
                    </div>
                    <button
                        className="deploy-cam-btn"
                        onClick={() => setShowAddModal(true)}
                    >
                        <Plus size={18} />
                        <span>Deploy CCTV Node in {activeZone.id}</span>
                    </button>
                </div>

                <div className="zone-summary-card">
                    <div className="zone-summary-left">
                        <h2>{activeZone.name}</h2>
                        <p>{activeZone.description || 'High-density commercial and transit junction equipped with real-time biometric video surveillance and automated alert broadcasting.'}</p>
                    </div>
                    <div className="zone-summary-right">
                        <div className="zone-metric-box">
                            <span className="metric-label">Sector Radius</span>
                            <span className="metric-value">{Math.round((activeZone.radius || 4000)/1000)} km</span>
                        </div>
                        <div className="zone-metric-box">
                            <span className="metric-label">Sector Center GPS</span>
                            <span className="metric-value" style={{ fontSize: '1rem', fontFamily: 'monospace' }}>
                                {activeZone.center[0].toFixed(4)}° N, {activeZone.center[1].toFixed(4)}° E
                            </span>
                        </div>
                        <div className="zone-metric-box">
                            <span className="metric-label">Operational Status</span>
                            <span className="metric-value" style={{ color: '#4CAF50' }}>ACTIVE</span>
                        </div>
                    </div>
                </div>

                <section className="cctv-grid-section">
                    <div className="cctv-grid-header">
                        <h3>Deployed Camera Nodes in {activeZone.name.split('(')[0]} ({activeZoneCameras.length})</h3>
                    </div>

                    <div className="cameras-grid">
                        {activeZoneCameras.length === 0 ? (
                            <div style={{ color: '#888', padding: '2rem 0' }}>
                                No surveillance nodes currently deployed in this sector. Use "Deploy CCTV Node" to add camera hardware.
                            </div>
                        ) : (
                            activeZoneCameras.map((cam) => {
                                const gaitCam = getGaitCamera(cam.id);
                                const loading = Boolean(actionLoading[cam.id]);
                                const error = actionErrors[cam.id];

                                const isConnected = Boolean(gaitCam && (gaitCam.status === 'ACTIVE' || gaitCam.status === 'connected') && (gaitCam.processed_frames > 0 || (gaitCam.fps && gaitCam.fps > 0)));
                                const workerActive = Boolean(gaitCam && (gaitCam.status === 'ACTIVE' || gaitCam.status === 'connected'));

                                const rawSourceType = isConnected ? (gaitCam?.source_type || gaitCam?.resolved_source_type || null) : null;
                                const isWebcam = rawSourceType ? (String(rawSourceType).toLowerCase() === 'webcam' || String(rawSourceType).toLowerCase() === 'usb') : false;
                                const detectedSourceLabel = rawSourceType ? (isWebcam ? 'Webcam' : 'RTSP') : null;

                                let statusLabel = 'Standby';
                                let statusColor = 'var(--text-muted)';
                                let statusDotColor = '#888';

                                if (loading) {
                                    statusLabel = 'Connecting...';
                                    statusColor = '#00E5FF';
                                    statusDotColor = '#00E5FF';
                                } else if (isConnected) {
                                    statusLabel = 'Connected';
                                    statusColor = '#06D6A0';
                                    statusDotColor = '#06D6A0';
                                } else if (error) {
                                    statusLabel = 'Connection Failed';
                                    statusColor = '#FF6B6B';
                                    statusDotColor = '#FF6B6B';
                                } else if (cam.status === 'OFFLINE') {
                                    statusLabel = 'Disconnected';
                                    statusColor = 'var(--text-muted)';
                                    statusDotColor = '#888';
                                } else {
                                    statusLabel = 'Standby';
                                    statusColor = 'var(--text-muted)';
                                    statusDotColor = '#888';
                                }

                                return (
                                    <div key={cam.id} className="cctv-camera-card">
                                        <div className="cam-card-header">
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                                <span className="cam-id-tag">{cam.id}</span>
                                                {isConnected && detectedSourceLabel && (
                                                    <span className={`cam-source-badge ${isWebcam ? 'source-webcam' : 'source-rtsp'}`}>
                                                        {isWebcam ? <Video size={12} /> : <Radio size={12} />}
                                                        Source: {detectedSourceLabel}
                                                    </span>
                                                )}
                                            </div>
                                            <span className="cam-status-indicator" style={{ color: statusColor }}>
                                                <span className="status-dot" style={{ background: statusDotColor }}></span>
                                                Status: {statusLabel}
                                            </span>
                                        </div>

                                        <div className="cam-card-body">
                                            <h4>{cam.name}</h4>

                                            <LiveCameraFeed
                                                cameraId={cam.id}
                                                cameraName={cam.name}
                                                workerActive={workerActive}
                                                telemetry={gaitCam}
                                            />

                                            <div className="cam-specs-list">
                                                {isConnected && detectedSourceLabel && (
                                                    <div className="spec-row">
                                                        <span>Source</span>
                                                        <span className="spec-val" style={{ color: isWebcam ? '#00E5FF' : '#5CE1E6', fontWeight: 700 }}>
                                                            {detectedSourceLabel} {gaitCam?.resolved_source_label ? `(${gaitCam.resolved_source_label})` : ''}
                                                        </span>
                                                    </div>
                                                )}
                                                <div className="spec-row">
                                                    <span>Status</span>
                                                    <span className="spec-val" style={{ color: statusColor, fontWeight: 700 }}>
                                                        {statusLabel}
                                                    </span>
                                                </div>
                                                <div className="spec-row">
                                                    <span>Resolution / Stream</span>
                                                    <span className="spec-val">{cam.resolution}</span>
                                                </div>
                                                <div className="spec-row">
                                                    <span>Coordinates</span>
                                                    <span className="spec-val" style={{ fontFamily: 'monospace' }}>
                                                        {cam.lat ? cam.lat.toFixed(4) : '0.0000'}°, {cam.lng ? cam.lng.toFixed(4) : '0.0000'}°
                                                    </span>
                                                </div>
                                                <div className="spec-row">
                                                    <span>Backend Worker</span>
                                                    <span className="spec-val" style={{ color: isConnected ? '#06D6A0' : 'var(--text-muted)' }}>
                                                        {isConnected ? `🟢 Active (${gaitCam?.fps || 15} FPS)` : (loading ? '🟡 Connecting...' : '⚪ Standby')}
                                                    </span>
                                                </div>
                                            </div>

                                            <div className="cam-card-actions">
                                                <button
                                                    className="trigger-alarm-btn"
                                                    onClick={() => handleSimulateAlert(cam)}
                                                >
                                                    <AlertTriangle size={16} />
                                                    <span>Simulate Pursuit Alert</span>
                                                </button>

                                                <div style={{ display: 'flex', gap: '0.5rem' }}>
                                                    <button
                                                        className="stream-details-btn"
                                                        style={{ flex: 1 }}
                                                        onClick={() => setSelectedCamStream(cam)}
                                                    >
                                                        Live Stream Details
                                                    </button>

                                                    <button
                                                        className="stream-details-btn"
                                                        style={{
                                                            background: workerActive ? 'rgba(255, 107, 107, 0.2)' : 'rgba(0, 229, 255, 0.2)',
                                                            borderColor: workerActive ? '#FF6B6B' : '#00E5FF',
                                                            color: workerActive ? '#FF6B6B' : '#00E5FF',
                                                            display: 'flex',
                                                            alignItems: 'center',
                                                            justifyContent: 'center',
                                                            gap: '0.3rem',
                                                        }}
                                                        disabled={loading}
                                                        onClick={() => handleToggleGaitWorker(cam)}
                                                    >
                                                        {loading ? (
                                                            <Loader size={14} className="activity-spin" />
                                                        ) : workerActive ? (
                                                            <Square size={14} />
                                                        ) : (
                                                            <Play size={14} />
                                                        )}
                                                        <span>{loading ? 'Processing...' : workerActive ? 'Stop Stream' : 'Start Stream'}</span>
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                );
                            })
                        )}
                    </div>
                </section>
            </main>

            {selectedCamStream && (
                <div className="cctv-modal-overlay">
                    <div className="cctv-modal-card" style={{ maxWidth: '680px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border)', paddingBottom: '0.75rem' }}>
                            <h3 style={{ margin: 0 }}>Sentinel Live Stream: {selectedCamStream.id}</h3>
                            <button onClick={() => setSelectedCamStream(null)} style={{ background: 'transparent', border: 'none', color: '#fff', cursor: 'pointer' }}>
                                <X size={24} />
                            </button>
                        </div>

                        <div style={{ padding: '1rem 0' }}>
                            <div style={{ marginBottom: '1rem', borderRadius: '8px', overflow: 'hidden', border: '2px solid #00E5FF' }}>
                                <LiveCameraFeed
                                    cameraId={selectedCamStream.id}
                                    cameraName={selectedCamStream.name}
                                    workerActive={isGaitWorkerActive(selectedCamStream.id)}
                                    telemetry={getGaitCamera(selectedCamStream.id)}
                                />
                            </div>

                            {streamEvents.length > 0 && (
                                <div style={{ background: 'rgba(0,0,0,0.4)', padding: '0.75rem', borderRadius: '8px', border: '1px solid var(--border)', marginBottom: '1rem', maxHeight: '140px', overflowY: 'auto' }}>
                                    <div style={{ fontSize: '0.75rem', fontWeight: 800, color: 'var(--sky)', marginBottom: '0.4rem', textTransform: 'uppercase' }}>
                                        Live Recognition Telemetry Stream
                                    </div>
                                    {streamEvents.map(evt => (
                                        <div key={evt.event_id} style={{ fontSize: '0.8rem', display: 'flex', justifyContent: 'space-between', padding: '0.2rem 0', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                                            <span style={{ color: evt.decision === 'KNOWN' ? '#06D6A0' : evt.decision === 'UNCERTAIN' ? '#FFD166' : '#FF6B6B', fontWeight: 700 }}>
                                                {evt.identity} ({evt.decision})
                                            </span>
                                            <span style={{ fontFamily: 'monospace', color: '#A0E4E8' }}>{(evt.confidence * 100).toFixed(1)}% Match</span>
                                        </div>
                                    ))}
                                </div>
                            )}

                            <div style={{ fontSize: '0.9rem', lineHeight: '1.6', color: 'var(--text-secondary)' }}>
                                <div><strong>Landmark Installation:</strong> {selectedCamStream.name}</div>
                                <div><strong>Assigned Sector:</strong> {activeZone.name}</div>
                                <div><strong>Detected Source:</strong> <span style={{ color: isGaitWorkerActive(selectedCamStream.id) && getGaitCamera(selectedCamStream.id)?.source_type ? '#00E5FF' : 'var(--text-muted)', fontWeight: 700 }}>{isGaitWorkerActive(selectedCamStream.id) && getGaitCamera(selectedCamStream.id)?.source_type ? String(getGaitCamera(selectedCamStream.id)?.source_type || '').toUpperCase() : 'STANDBY (NOT CONNECTED)'}</span></div>
                                <div><strong>Hardware Source Endpoint:</strong> <code>{isGaitWorkerActive(selectedCamStream.id) ? (getGaitCamera(selectedCamStream.id)?.source || selectedCamStream.ip || 'Auto-Detected Device') : 'Standby'}</code></div>
                                <div><strong>Exact GPS Placement:</strong> <code>{selectedCamStream.lat ? selectedCamStream.lat.toFixed(4) : '0.0000'}, {selectedCamStream.lng ? selectedCamStream.lng.toFixed(4) : '0.0000'}</code></div>
                                <div><strong>Connection Status:</strong> <code style={{ color: isGaitWorkerActive(selectedCamStream.id) ? '#06D6A0' : '#888' }}>{isGaitWorkerActive(selectedCamStream.id) ? 'CONNECTED (ACTIVE)' : 'STANDBY'}</code></div>
                                <div><strong>Active Stream Consumers:</strong> <code>{getGaitCamera(selectedCamStream.id)?.active_clients ?? 0}</code></div>
                            </div>
                        </div>

                        <div className="modal-actions-cctv" style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end' }}>
                            <button
                                className="btn-auto-gps"
                                style={{ background: isGaitWorkerActive(selectedCamStream.id) ? 'rgba(255,107,107,0.2)' : 'rgba(0,229,255,0.2)', borderColor: isGaitWorkerActive(selectedCamStream.id) ? '#FF6B6B' : '#00E5FF', color: isGaitWorkerActive(selectedCamStream.id) ? '#FF6B6B' : '#00E5FF' }}
                                onClick={() => handleToggleGaitWorker(selectedCamStream)}
                            >
                                {isGaitWorkerActive(selectedCamStream.id) ? 'Stop Live Stream' : 'Start Live Stream'}
                            </button>
                            <button className="btn-submit-cctv" onClick={() => setSelectedCamStream(null)}>
                                Close Stream Window
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {showAddModal && (
                <div className="cctv-modal-overlay">
                    <form className="cctv-modal-card" onSubmit={handleAddCameraSubmit}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', borderBottom: '1px solid var(--border)', paddingBottom: '0.75rem' }}>
                            <h3 style={{ margin: 0, border: 'none', padding: 0 }}>Deploy Custom CCTV Node ({activeZone.id})</h3>
                            <button type="button" onClick={() => setShowAddModal(false)} style={{ background: 'transparent', border: 'none', color: '#fff', cursor: 'pointer' }}>
                                <X size={22} />
                            </button>
                        </div>

                        <div className="form-group-cctv">
                            <label>Node Reference Identifier</label>
                            <input type="text" value={newCam.id} disabled style={{ opacity: 0.7 }} />
                        </div>

                        <div className="form-group-cctv">
                            <label>Camera Landmark / Location Name</label>
                            <input
                                type="text"
                                placeholder="e.g. Town Hall Intersection Gate 4"
                                value={newCam.name}
                                onChange={(e) => setNewCam({...newCam, name: e.target.value})}
                                required
                            />
                        </div>

                        <div className="form-group-cctv">
                            <div className="gps-detect-row">
                                <label>GPS Installation Coordinates</label>
                                <button type="button" className="btn-auto-gps" onClick={handleAutoDetectGPS} disabled={isDetectingGPS}>
                                    <MapPin size={14} />
                                    <span>{isDetectingGPS ? 'Detecting GPS...' : 'Auto-Detect Current GPS'}</span>
                                </button>
                            </div>
                            <div className="coords-row-cctv">
                                <input
                                    type="number"
                                    step="0.000001"
                                    placeholder="Latitude (e.g. 6.9271)"
                                    value={newCam.lat}
                                    onChange={(e) => setNewCam({...newCam, lat: e.target.value})}
                                    required
                                />
                                <input
                                    type="number"
                                    step="0.000001"
                                    placeholder="Longitude (e.g. 79.8612)"
                                    value={newCam.lng}
                                    onChange={(e) => setNewCam({...newCam, lng: e.target.value})}
                                    required
                                />
                            </div>
                        </div>

                        <div className="form-group-cctv">
                            <label>Stream & Recognition Capabilities</label>
                            <select
                                value={newCam.resolution}
                                onChange={(e) => setNewCam({...newCam, resolution: e.target.value})}
                            >
                                <option value="4K IR PTZ Stream & AI Face Recognition">4K IR PTZ Stream & AI Face Recognition</option>
                                <option value="1080p LPR (License Plate) & Facial Analytics">1080p LPR (License Plate) & Facial Analytics</option>
                                <option value="Thermal Infrared & Biometric Motion Tracking">Thermal Infrared & Biometric Motion Tracking</option>
                                <option value="Ultra-HD Omni-Directional 360° AI Feed">Ultra-HD Omni-Directional 360° AI Feed</option>
                            </select>
                        </div>

                        <div className="form-group-cctv">
                            <label>Assigned IP Endpoint</label>
                            <input
                                type="text"
                                value={newCam.ip}
                                onChange={(e) => setNewCam({...newCam, ip: e.target.value})}
                            />
                        </div>

                        <div className="modal-actions-cctv">
                            <button type="button" className="btn-cancel-cctv" onClick={() => setShowAddModal(false)}>
                                Cancel
                            </button>
                            <button type="submit" className="btn-submit-cctv">
                                🚀 Confirm Node Deployment
                            </button>
                        </div>
                    </form>
                </div>
            )}
        </div>
    );
};

export default CctvNetwork;
