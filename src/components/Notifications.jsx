import React, { useState, useEffect } from 'react';
import { Trash2, XCircle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import './Notifications.css';

const Notifications = ({ isOpen, onClose }) => {
    const navigate = useNavigate();
    const { currentUser } = useAuth();
    
    const email = currentUser?.email || '';
    const isAdmin = email.toLowerCase().includes('admin');

    const adminNotifs = [
        { id: 1, title: '[SYSTEM] Policies Compiled', details: 'Operational rules successfully written to all nodes.', caseId: null },
        { id: 2, title: '[SURVEILLANCE] CCTV Feed Link Drops', details: 'Camera 04 dropped connection handshake.', caseId: null },
        { id: 3, title: '[DATABASE] Hot Storage Snapshot Complete', details: 'Backup backup_index_1029 successfully archived.', caseId: null },
        { id: 4, title: '[AUTH] New Account Authorized', details: 'Investigator user "inv_agentX" registered.', caseId: null }
    ];

    const invNotifs = [
        { id: 1, title: 'BIOMETRIC ALERT: Case ID #9103 Match', details: '94% facial match detected on Core Server Vault CCTV.', caseId: '9103' },
        { id: 2, title: 'NEW CASE REPORTED: Sector Alpha', details: 'Missing person case reported for Alice Smith.', caseId: 'Alice-Smith' },
        { id: 3, title: 'CASE STATUS UPDATE: Case ID #8172', details: 'Case status changed to INVESTIGATING.', caseId: '8172' }
    ];

    const [notifications, setNotifications] = useState([]);

    useEffect(() => {
        if (isOpen) {
            setNotifications(isAdmin ? adminNotifs : invNotifs);
        }
    }, [isOpen, isAdmin]);

    const handleClearNotifications = () => {
        setNotifications([]);
    };

    const handleNotificationClick = (caseId) => {
        if (!caseId) return; // System logs don't navigate to case pages
        onClose(); 
        navigate(`/case/${caseId}`);
    };

    if (!isOpen) return null;

    return (
        <div className="notification-overlay" onClick={onClose}>
            <div className="notification-modal" onClick={(e) => e.stopPropagation()}>
                <div className="notif-header">
                    <h2>Notifications</h2>
                    <div className="notif-actions">
                        <button className="notif-clear-btn" onClick={handleClearNotifications} title="Clear all">
                            <Trash2 size={22} />
                        </button>
                        <button className="notif-close-btn" onClick={onClose} title="Close">
                            <XCircle size={26} fill="#FF6B6B" color="rgba(35,37,40,0.9)" />
                        </button>
                    </div>
                </div>
                
                <div className="notif-list">
                    {notifications.length === 0 ? (
                        <p className="notif-empty">No new notifications.</p>
                    ) : (
                        notifications.map((notif) => (
                            <div 
                                key={notif.id}
                                className="notif-item" 
                                onClick={() => handleNotificationClick(notif.caseId)}
                                style={{ cursor: notif.caseId ? 'pointer' : 'default' }}
                            >
                                <p className="notif-title" style={{ fontWeight: '700', color: 'var(--mist)' }}>{notif.title}</p>
                                <p className="notif-details" style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: '2px' }}>{notif.details}</p>
                                {notif.caseId && <span className="notif-case-tag" style={{ fontSize: '0.65rem', color: 'var(--sky)', marginTop: '4px', display: 'inline-block' }}>Case: {notif.caseId}</span>}
                            </div>
                        ))
                    )}
                </div>
            </div>
        </div>
    );
};

export default Notifications;
