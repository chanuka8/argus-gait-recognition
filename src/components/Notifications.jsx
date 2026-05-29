import React, { useState } from 'react';
import { Trash2, XCircle } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import './Notifications.css';

const Notifications = ({ isOpen, onClose }) => {
    const navigate = useNavigate();
    
    
    const [notifications, setNotifications] = useState([]);

    const handleClearNotifications = () => {
        setNotifications([]);
    };

    const handleNotificationClick = (caseId) => {
        onClose(); 
        navigate(`/case/${caseId || '123'}`);
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
                            <XCircle size={26} fill="#FF5252" color="rgba(4,6,84,0.9)" />
                        </button>
                    </div>
                </div>
                
                <div className="notif-list">
                    {notifications.length === 0 ? (
                        <p className="notif-empty">No new notifications.</p>
                    ) : (
                        notifications.map((notif) => (
                            <React.Fragment key={notif.id}>
                                <div 
                                    className="notif-item" 
                                    onClick={() => handleNotificationClick(notif.caseId)}
                                    style={{ cursor: 'pointer' }}
                                >
                                    {notif.title ? (
                                        <p className="notif-title">{notif.title}</p>
                                    ) : (
                                        <>
                                            <p className="notif-case">Case id : {notif.caseId || '______'}</p>
                                            <p className="notif-details">{notif.details}</p>
                                        </>
                                    )}
                                </div>
                            </React.Fragment>
                        ))
                    )}
                </div>
            </div>
        </div>
    );
};

export default Notifications;
