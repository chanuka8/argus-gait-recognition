import React, { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, User as UserIcon, Bell } from 'lucide-react';
import logo from '../assets/logo.png';
import Notifications from './Notifications';
import UserProfileModal from './UserProfileModal';
import './CaseDetails.css';

const CaseDetails = () => {
    const { id } = useParams();
    const navigate = useNavigate();
    const [showNotifications, setShowNotifications] = useState(false);
    const [showProfile, setShowProfile] = useState(false);

    const displayId = id && id !== 'undefined' ? id : '_______________';

    return (
        <div className="case-details-page">
            <Notifications isOpen={showNotifications} onClose={() => setShowNotifications(false)} />
            <UserProfileModal isOpen={showProfile} onClose={() => setShowProfile(false)} />
            
            <header className="case-details-header">
                <div className="header-left">
                    <button className="case-back-btn" onClick={() => navigate(-1)}>
                        <ArrowLeft size={24} color="#000" />
                    </button>
                    <img src={logo} alt="Argus Logo" className="header-logo" />
                    <span className="header-title">ARGUS</span>
                </div>
                <div className="header-right">
                    <div className="user-profile" onClick={() => setShowProfile(true)} style={{ cursor: 'pointer' }}>
                        <UserIcon size={22} fill="#d6e4ea" color="#d6e4ea" />
                        <span>John Doe</span>
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
                    <h2 className="case-id-header">Case ID : {displayId}</h2>

                    <div className="case-layout">
                        <div className="case-info-panel">
                            <div className="case-icon-wrapper">
                                <UserIcon size={140} color="#a0e4e8" fill="#4ab8bd" />
                            </div>
                            
                            <div className="info-grid">
                                <div className="info-row">
                                    <span className="info-label">Case type :</span>
                                    <span className="info-value">Abduction</span>
                                </div>
                                <div className="info-row">
                                    <span className="info-label">Case status :</span>
                                    <span className="info-value">Active</span>
                                </div>
                                <div className="info-row">
                                    <span className="info-label">Name :</span>
                                    <span className="info-value">Alex Smith</span>
                                </div>
                                <div className="info-row">
                                    <span className="info-label">Gender :</span>
                                    <span className="info-value">Male</span>
                                </div>
                                <div className="info-row">
                                    <span className="info-label">NIC :</span>
                                    <span className="info-value">0123456789012</span>
                                </div>
                                <div className="info-row">
                                    <span className="info-label">Age :</span>
                                    <span className="info-value">24</span>
                                </div>
                                
                                <div className="about-case">
                                    <span className="info-label">About Case :</span>
                                    <p className="about-text">
                                        Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua
                                    </p>
                                </div>
                            </div>
                        </div>

                        <div className="case-visuals-panel">
                            <div className="case-map-box"></div>
                            
                            <div className="case-feed-box">
                                <p className="feed-placeholder">Live Camera Feed / Surveillance Footage</p>
                            </div>

                            <div className="case-actions">
                                <button className="case-action-btn">Update Status</button>
                                <button className="case-action-btn">Close Case</button>
                            </div>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
};

export default CaseDetails;
