import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, User as UserIcon, Bell, Loader, X, Edit, ShieldAlert, MapPin } from 'lucide-react';
import logo from '../assets/logo.png';
import Notifications from './Notifications';
import UserProfileModal from './UserProfileModal';
import { db } from '../firebaseConfig';
import { doc, getDoc, updateDoc, collection, onSnapshot, query, where } from 'firebase/firestore';
import { useAuth } from '../contexts/AuthContext';
import { addLog } from '../utils/logService';
import { MapContainer, TileLayer, Marker, Popup, Circle, Polyline } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import './CaseDetails.css';

const createCaseDetectionIcon = () => {
    return L.divIcon({
        className: 'custom-detection-marker',
        html: `<div style="
            background-color: #FF6F00;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            border: 3px solid #FFF;
            box-shadow: 0 0 0 8px rgba(255, 111, 0, 0.45), 0 2px 10px rgba(0,0,0,0.8);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 13px;
        ">
            🚨
        </div>`,
        iconSize: [30, 30],
        iconAnchor: [15, 15],
        popupAnchor: [0, -16]
    });
};

const createCaseDetailsIcon = (status) => {
    let color = '#E53935';
    let pulseColor = 'rgba(229, 57, 53, 0.4)';
    if (status?.toLowerCase() === 'found' || status?.toLowerCase() === 'closed') {
        color = '#4CAF50';
        pulseColor = 'rgba(76, 175, 80, 0.4)';
    } else if (status?.toLowerCase() === 'cold') {
        color = '#42A5F5';
        pulseColor = 'rgba(66, 165, 245, 0.4)';
    }

    return L.divIcon({
        className: 'case-details-marker',
        html: `<div style="
            background-color: ${color};
            width: 28px;
            height: 28px;
            border-radius: 50%;
            border: 3px solid #ffffff;
            box-shadow: 0 0 0 8px ${pulseColor}, 0 2px 8px rgba(0,0,0,0.6);
            display: flex;
            align-items: center;
            justify-content: center;
        ">
            <div style="width: 10px; height: 10px; background: #fff; border-radius: 50%;"></div>
        </div>`,
        iconSize: [28, 28],
        iconAnchor: [14, 14],
        popupAnchor: [0, -18]
    });
};

const CaseDetails = () => {
    const { id } = useParams();
    const navigate = useNavigate();
    const { currentUser } = useAuth();
    
    const [showNotifications, setShowNotifications] = useState(false);
    const [showProfile, setShowProfile] = useState(false);

    // Live case states
    const [caseData, setCaseData] = useState(null);
    const [mediaData, setMediaData] = useState(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');

    // Status modal states
    const [showStatusModal, setShowStatusModal] = useState(false);
    const [selectedStatus, setSelectedStatus] = useState('Investigating');
    const [isSavingStatus, setIsSavingStatus] = useState(false);
    const [searchRadius, setSearchRadius] = useState(5000);
    const [caseDetections, setCaseDetections] = useState([]);

    const displayId = id && id !== 'undefined' ? id : '_______________';

    const getCaseCoords = () => {
        if (caseData && caseData.lastSeenLocation && caseData.lastSeenLocation.lat && caseData.lastSeenLocation.lng) {
            return [parseFloat(caseData.lastSeenLocation.lat), parseFloat(caseData.lastSeenLocation.lng)];
        }
        if (caseData && caseData.latitude && caseData.longitude) {
            return [parseFloat(caseData.latitude), parseFloat(caseData.longitude)];
        }
        return [6.9271, 79.8612];
    };

    const getCaseLocationLabel = () => {
        if (caseData && caseData.lastSeenLocation && caseData.lastSeenLocation.name) {
            return caseData.lastSeenLocation.name;
        }
        if (caseData && caseData.locationName) return caseData.locationName;
        return "Unspecified Geolocation";
    };

    const caseCoords = getCaseCoords();
    const caseLocLabel = getCaseLocationLabel();
    const markerIcon = createCaseDetailsIcon(caseData?.status);

    useEffect(() => {
        if (!id || id === 'undefined') return;
        try {
            const q = query(collection(db, 'detections'), where('caseId', '==', id));
            const unsubscribe = onSnapshot(q, (snapshot) => {
                const list = [];
                snapshot.forEach(doc => {
                    const data = doc.data();
                    if (data.coordinates && data.coordinates.lat && data.coordinates.lng) {
                        list.push({ id: doc.id, ...data });
                    }
                });
                setCaseDetections(list);
            });
            return () => unsubscribe();
        } catch (err) {
            console.error("Error subscribing to case detections:", err);
        }
    }, [id]);

    useEffect(() => {
        const fetchCaseDetails = async () => {
            if (!id || id === 'undefined') {
                setError('Invalid case reference identifier.');
                setIsLoading(false);
                return;
            }

            try {
                setIsLoading(true);
                setError('');
                
                // 1. Fetch victim document details
                const victimRef = doc(db, 'victims', id);
                const victimSnap = await getDoc(victimRef);
                
                if (victimSnap.exists()) {
                    const data = victimSnap.data();
                    setCaseData(data);
                    setSelectedStatus(data.status || 'Investigating');
                } else {
                    setError('Case record not found in system database.');
                }

                // 2. Fetch media attachments
                const mediaRef = doc(db, 'person_media', id);
                const mediaSnap = await getDoc(mediaRef);
                if (mediaSnap.exists()) {
                    setMediaData(mediaSnap.data());
                }

            } catch (err) {
                console.error('Error fetching case logs:', err);
                setError('Failed to connect to database feed.');
            } finally {
                setIsLoading(false);
            }
        };

        fetchCaseDetails();
    }, [id]);

    const handleUpdateStatusSubmit = async (e) => {
        e.preventDefault();
        setIsSavingStatus(true);

        try {
            const victimRef = doc(db, 'victims', id);
            await updateDoc(victimRef, { status: selectedStatus });
            
            setCaseData(prev => ({ ...prev, status: selectedStatus }));
            
            // Record status change in logs
            addLog(
                'info',
                `Case ${displayId} status updated to ${selectedStatus}`,
                `Investigator @${currentUser?.username || 'unknown'} updated case status to ${selectedStatus}.`,
                currentUser?.username || 'unknown'
            );

            setShowStatusModal(false);
        } catch (err) {
            console.error('Error updating case status:', err);
            alert('Failed to update status: ' + (err.message || err.toString()));
        } finally {
            setIsSavingStatus(false);
        }
    };

    const handleCloseCase = async () => {
        if (window.confirm(`Are you sure you want to CLOSE case ${displayId}? This will mark the investigation as complete.`)) {
            setIsSavingStatus(true);
            try {
                const victimRef = doc(db, 'victims', id);
                await updateDoc(victimRef, { status: 'Closed' });
                
                setCaseData(prev => ({ ...prev, status: 'Closed' }));
                setSelectedStatus('Closed');

                // Record closure in logs
                addLog(
                    'warning',
                    `Case ${displayId} closed`,
                    `Investigator @${currentUser?.username || 'unknown'} permanently closed the case log.`,
                    currentUser?.username || 'unknown'
                );

            } catch (err) {
                console.error('Error closing case:', err);
                alert('Failed to close case: ' + (err.message || err.toString()));
            } finally {
                setIsSavingStatus(false);
            }
        }
    };

    return (
        <div className="case-details-page">
            <Notifications isOpen={showNotifications} onClose={() => setShowNotifications(false)} />
            <UserProfileModal isOpen={showProfile} onClose={() => setShowProfile(false)} />
            
            <header className="case-details-header">
                <div className="header-left">
                    <button className="case-back-btn" onClick={() => navigate(-1)} title="Go Back">
                        <ArrowLeft size={20} color="#fff" />
                    </button>
                    <img src={logo} alt="Argus Logo" className="header-logo" />
                    <span className="header-title">ARGUS</span>
                </div>
                <div className="header-right">
                    <div className="user-profile" onClick={() => setShowProfile(true)} style={{ cursor: 'pointer' }}>
                        <UserIcon size={22} fill="#d6e4ea" color="#d6e4ea" />
                        <span>{currentUser?.username || 'John Doe'}</span>
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

            <main className="case-details-content">
                <div className="case-details-container">
                    {isLoading ? (
                        <div className="case-loading-box">
                            <Loader className="spinner" size={40} color="var(--sky)" />
                            <p>Loading database case logs...</p>
                        </div>
                    ) : error ? (
                        <div className="case-error-box">
                            <ShieldAlert size={48} color="var(--status-missing)" />
                            <h2>Query Error</h2>
                            <p>{error}</p>
                            <button className="case-back-btn" style={{ borderRadius: '4px', width: 'auto', height: 'auto', padding: '0.5rem 1.5rem', marginTop: '1rem' }} onClick={() => navigate(-1)}>
                                Return to Dashboard
                            </button>
                        </div>
                    ) : (
                        <>
                            <h2 className="case-id-header">Case ID : {displayId}</h2>

                            <div className="case-layout">
                                <div className="case-info-panel">
                                    <div className="case-icon-wrapper">
                                        <UserIcon size={140} color="#a0e4e8" fill="#4ab8bd" />
                                    </div>
                                    
                                    <div className="info-grid">
                                        <div className="info-row">
                                            <span className="info-label">Case type :</span>
                                            <span className="info-value">{caseData?.caseType || 'N/A'}</span>
                                        </div>
                                        <div className="info-row">
                                            <span className="info-label">Case status :</span>
                                            <span className="info-value" style={{ textTransform: 'capitalize' }}>
                                                {caseData?.status || 'N/A'}
                                            </span>
                                        </div>
                                        <div className="info-row">
                                            <span className="info-label">Name :</span>
                                            <span className="info-value">{caseData?.name || 'N/A'}</span>
                                        </div>
                                        <div className="info-row">
                                            <span className="info-label">Gender :</span>
                                            <span className="info-value">{caseData?.gender || 'N/A'}</span>
                                        </div>
                                        <div className="info-row">
                                            <span className="info-label">NIC :</span>
                                            <span className="info-value" style={{ fontFamily: 'monospace' }}>{caseData?.nic || 'N/A'}</span>
                                        </div>
                                        <div className="info-row">
                                            <span className="info-label">Age :</span>
                                            <span className="info-value">{caseData?.age || 'N/A'}</span>
                                        </div>
                                        <div className="info-row">
                                            <span className="info-label">Last seen :</span>
                                            <span className="info-value" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                                <MapPin size={14} color="var(--sky)" /> {caseLocLabel}
                                            </span>
                                        </div>
                                        
                                        <div className="about-case">
                                            <span className="info-label">About Case :</span>
                                            <p className="about-text">
                                                This case file contains records for victim {caseData?.name || 'N/A'} with classification '{caseData?.caseType || 'N/A'}'. Logged into system databases under status '{caseData?.status || 'N/A'}'.
                                            </p>
                                        </div>
                                    </div>
                                </div>

                                <div className="case-visuals-panel">
                                    <div className="case-map-box">
                                        <div className="radius-controls-overlay">
                                            <span className="radius-label">Search Zone:</span>
                                            <button 
                                                type="button"
                                                className={`radius-pill ${searchRadius === 2000 ? 'active' : ''}`}
                                                onClick={() => setSearchRadius(2000)}
                                            >
                                                2km
                                            </button>
                                            <button 
                                                type="button"
                                                className={`radius-pill ${searchRadius === 5000 ? 'active' : ''}`}
                                                onClick={() => setSearchRadius(5000)}
                                            >
                                                5km
                                            </button>
                                            <button 
                                                type="button"
                                                className={`radius-pill ${searchRadius === 10000 ? 'active' : ''}`}
                                                onClick={() => setSearchRadius(10000)}
                                            >
                                                10km
                                            </button>
                                            <button 
                                                type="button"
                                                className={`radius-pill ${searchRadius === 25000 ? 'active' : ''}`}
                                                onClick={() => setSearchRadius(25000)}
                                            >
                                                25km
                                            </button>
                                        </div>
                                        <MapContainer 
                                            key={caseCoords.join(',')}
                                            center={caseCoords} 
                                            zoom={12} 
                                            scrollWheelZoom={true}
                                            style={{ height: '100%', width: '100%' }}
                                        >
                                            <TileLayer
                                                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                                                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                                            />
                                            <Circle 
                                                center={caseCoords} 
                                                radius={searchRadius} 
                                                pathOptions={{ 
                                                    color: caseData?.status?.toLowerCase() === 'found' || caseData?.status?.toLowerCase() === 'closed' ? '#4CAF50' : 
                                                           caseData?.status?.toLowerCase() === 'cold' ? '#42A5F5' : '#E53935', 
                                                    fillColor: caseData?.status?.toLowerCase() === 'found' || caseData?.status?.toLowerCase() === 'closed' ? '#4CAF50' : 
                                                               caseData?.status?.toLowerCase() === 'cold' ? '#42A5F5' : '#E53935', 
                                                    fillOpacity: 0.15, 
                                                    weight: 2,
                                                    dashArray: '4, 4'
                                                }} 
                                            />
                                            <Marker position={caseCoords} icon={markerIcon}>
                                                <Popup className="argus-custom-popup">
                                                    <div style={{ padding: '0.2rem', color: '#111', fontFamily: 'sans-serif' }}>
                                                        <strong style={{ fontSize: '0.95rem' }}>{caseData?.name || 'Subject'}</strong>
                                                        <div style={{ marginTop: '0.25rem', fontSize: '0.85rem' }}>
                                                            <div><strong>Last Known:</strong> {caseLocLabel}</div>
                                                            <div><strong>Search Perimeter:</strong> {searchRadius / 1000} km</div>
                                                        </div>
                                                    </div>
                                                </Popup>
                                            </Marker>
                                            {caseDetections.map((det, dIdx) => {
                                                const detCoords = [parseFloat(det.coordinates.lat), parseFloat(det.coordinates.lng)];
                                                if (isNaN(detCoords[0]) || isNaN(detCoords[1])) return null;
                                                const detIcon = createCaseDetectionIcon();
                                                return (
                                                    <React.Fragment key={`det-${det.id || dIdx}`}>
                                                        <Polyline 
                                                            positions={[caseCoords, detCoords]} 
                                                            pathOptions={{ color: '#FF6F00', weight: 3, dashArray: '8, 8', opacity: 0.8 }} 
                                                        />
                                                        <Marker position={detCoords} icon={detIcon}>
                                                            <Popup className="argus-custom-popup">
                                                                <div style={{ padding: '0.2rem', color: '#111', fontFamily: 'sans-serif' }}>
                                                                    <strong style={{ color: '#FF6F00', fontSize: '0.9rem' }}>🚨 CAMERA SIGHTING LOG</strong>
                                                                    <div style={{ marginTop: '0.25rem', fontSize: '0.85rem' }}>
                                                                        <div><strong>Camera:</strong> {det.cameraId || 'CCTV Feed'}</div>
                                                                        <div><strong>Location:</strong> {det.locationName}</div>
                                                                    </div>
                                                                </div>
                                                            </Popup>
                                                        </Marker>
                                                    </React.Fragment>
                                                );
                                            })}
                                        </MapContainer>
                                    </div>
                                    
                                    <div className="case-feed-box">
                                        {mediaData && ((mediaData.imageUrls && mediaData.imageUrls.length > 0) || (mediaData.videoUrls && mediaData.videoUrls.length > 0)) ? (
                                            <div className="victim-media-display">
                                                <h3>Case Upload Media & Visual Attachments</h3>
                                                <div className="victim-media-scroll">
                                                    {mediaData.imageUrls && mediaData.imageUrls.map((url, i) => (
                                                        <div key={`img-${i}`} className="media-thumbnail-card">
                                                            <img src={url} alt={`Victim Attachment ${i + 1}`} />
                                                        </div>
                                                    ))}
                                                    {mediaData.videoUrls && mediaData.videoUrls.map((url, i) => (
                                                        <div key={`vid-${i}`} className="media-thumbnail-card video-card">
                                                            <video src={url} controls />
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        ) : (
                                            <p className="feed-placeholder">No surveillance media logs or upload footage connected to this case log yet.</p>
                                        )}
                                    </div>

                                    <div className="case-actions">
                                        <button 
                                            className="case-action-btn" 
                                            onClick={() => setShowStatusModal(true)}
                                            disabled={caseData?.status?.toLowerCase() === 'closed'}
                                            style={caseData?.status?.toLowerCase() === 'closed' ? { opacity: 0.4, cursor: 'not-allowed', filter: 'grayscale(1)' } : {}}
                                        >
                                            Update Status
                                        </button>
                                        <button 
                                            className="case-action-btn close-case-style" 
                                            onClick={handleCloseCase}
                                            disabled={caseData?.status?.toLowerCase() === 'closed'}
                                            style={caseData?.status?.toLowerCase() === 'closed' ? { opacity: 0.4, cursor: 'not-allowed', filter: 'grayscale(1)' } : {}}
                                        >
                                            Close Case
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </>
                    )}
                </div>
            </main>

            {/* Status Update Modal */}
            {showStatusModal && (
                <div 
                    className="status-modal-overlay"
                    onClick={(e) => {
                        if (e.target === e.currentTarget) {
                            setShowStatusModal(false);
                        }
                    }}
                >
                    <div className="status-modal-card">
                        <div className="modal-header">
                            <div className="header-title-flex">
                                <Edit size={20} className="modal-header-icon" />
                                <h2>Update Case Status</h2>
                            </div>
                            <button className="modal-close-btn" onClick={() => setShowStatusModal(false)} disabled={isSavingStatus}>
                                <X size={20} />
                            </button>
                        </div>

                        <form onSubmit={handleUpdateStatusSubmit} className="status-update-form">
                            <div className="form-field">
                                <label>Operational Classification Status</label>
                                <select 
                                    value={selectedStatus}
                                    onChange={(e) => setSelectedStatus(e.target.value)}
                                    disabled={isSavingStatus}
                                >
                                    <option value="Investigating">Investigating</option>
                                    <option value="Cold">Cold</option>
                                    <option value="Found">Found</option>
                                    <option value="Closed">Closed (Resolved)</option>
                                </select>
                            </div>

                            <div className="modal-actions" style={{ display: 'flex', justifyContent: 'flex-end', gap: '1rem', marginTop: '1.5rem' }}>
                                <button 
                                    type="button" 
                                    className="modal-cancel-btn" 
                                    onClick={() => setShowStatusModal(false)}
                                    disabled={isSavingStatus}
                                >
                                    Cancel
                                </button>
                                <button 
                                    type="submit" 
                                    className="modal-confirm-btn"
                                    disabled={isSavingStatus}
                                >
                                    {isSavingStatus ? 'Saving Status...' : 'Update Status'}
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
        </div>
    );
};

export default CaseDetails;
