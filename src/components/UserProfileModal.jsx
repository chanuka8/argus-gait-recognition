import React from 'react';
import { XCircle, LogOut } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import './UserProfileModal.css';

const UserProfileModal = ({ isOpen, onClose }) => {
    const navigate = useNavigate();
    const { logout } = useAuth();

    if (!isOpen) return null;

    const handleLogout = async () => {
        try {
            await logout();
            onClose();
            navigate('/');
        } catch (error) {
            console.error("Failed to log out", error);
        }
    };

    return (
        <div className="profile-overlay">
            <div className="profile-modal">
                <button className="profile-close-btn" onClick={onClose} title="Close">
                    <XCircle size={28} fill="#ef4444" color="#5a5a5a" />
                </button>
                
                <div className="profile-content">
                    <div className="profile-icon-wrapper">
                        {/* Custom SVG to match the large solid user icon from screenshot perfectly */}
                        <svg viewBox="0 0 24 24" fill="white" xmlns="http://www.w3.org/2000/svg" width="180" height="180">
                            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z"/>
                        </svg>
                    </div>

                    <div className="profile-details-container">
                        <div className="profile-details">
                            <p><strong>Name :</strong> John Doe</p>
                            <p><strong>Position :</strong> Investigator</p>
                        </div>
                    </div>

                    <button className="logout-btn" onClick={handleLogout}>
                        <LogOut size={32} strokeWidth={2.5} />
                        <span>Logout</span>
                    </button>
                </div>
            </div>
        </div>
    );
};

export default UserProfileModal;
