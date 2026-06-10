import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, User as UserIcon, Bell, Loader, X, Edit, ShieldAlert } from 'lucide-react';
import logo from '../assets/logo.png';
import Notifications from './Notifications';
import UserProfileModal from './UserProfileModal';
import { db } from '../firebaseConfig';
import { doc, getDoc, updateDoc } from 'firebase/firestore';
import { useAuth } from '../contexts/AuthContext';
import { addLog } from '../utils/logService';
import './CaseDetails.css';

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

    const displayId = id && id !== 'undefined' ? id : '_______________';

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
                                        
                                        <div className="about-case">
                                            <span className="info-label">About Case :</span>
                                            <p className="about-text">
                                                This case file contains records for victim {caseData?.name || 'N/A'} with classification '{caseData?.caseType || 'N/A'}'. Logged into system databases under status '{caseData?.status || 'N/A'}'.
                                            </p>
                                        </div>
                                    </div>
                                </div>

                                <div className="case-visuals-panel">
                                    <div className="case-map-box" title="Victim Incident Map Coordinates"></div>
                                    
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
