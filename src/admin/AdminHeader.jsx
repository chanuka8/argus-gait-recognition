import React, { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { User, Bell, LayoutDashboard, Users, Video, Shield, FileClock, ShieldAlert } from 'lucide-react';
import logo from '../assets/logo.png';
import Notifications from '../components/Notifications';
import UserProfileModal from '../components/UserProfileModal';
import { useAuth } from '../contexts/AuthContext';
import './AdminHeader.css';

const AdminHeader = () => {
    const navigate = useNavigate();
    const location = useLocation();
    const { currentUser } = useAuth();
    const [showNotifications, setShowNotifications] = useState(false);
    const [showProfile, setShowProfile] = useState(false);

    const username = currentUser?.username || 'Admin';

    const menuItems = [
        { path: '/admin/dashboard', label: 'Dashboard', icon: <LayoutDashboard size={16} /> },
        { path: '/admin/users', label: 'Users', icon: <Users size={16} /> },
        { path: '/admin/surveillance', label: 'Surveillance', icon: <Video size={16} /> },
        { path: '/admin/policies', label: 'Policies', icon: <Shield size={16} /> },
        { path: '/admin/logs', label: 'Logs', icon: <FileClock size={16} /> }
    ];

    return (
        <header className="admin-header-nav">
            <Notifications isOpen={showNotifications} onClose={() => setShowNotifications(false)} />
            <UserProfileModal isOpen={showProfile} onClose={() => setShowProfile(false)} />

            <div className="admin-header-left">
                <img src={logo} alt="Argus Logo" className="admin-header-logo" onClick={() => navigate('/admin/dashboard')} style={{ cursor: 'pointer' }} />
                <div className="admin-brand-container">
                    <span className="admin-header-title">ARGUS</span>
                    <span className="admin-badge">ADMIN PANEL</span>
                </div>
            </div>

            <nav className="admin-nav-links">
                {menuItems.map((item) => {
                    const isActive = location.pathname === item.path;
                    return (
                        <button
                            key={item.path}
                            className={`admin-nav-btn ${isActive ? 'active' : ''}`}
                            onClick={() => navigate(item.path)}
                        >
                            {item.icon}
                            <span>{item.label}</span>
                        </button>
                    );
                })}
            </nav>

            <div className="admin-header-right">
                <div className="admin-user-profile" onClick={() => setShowProfile(true)}>
                    <User size={18} fill="#a0e4e8" color="#a0e4e8" />
                    <span>{username}</span>
                </div>

                <Bell
                    size={22}
                    className="admin-notification-bell"
                    fill="#5ce1e6"
                    color="#5ce1e6"
                    onClick={() => setShowNotifications(true)}
                />
            </div>
        </header>
    );
};

export default AdminHeader;
