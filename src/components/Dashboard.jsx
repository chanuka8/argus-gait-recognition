import React, { useState, useEffect } from 'react';
import { User, Bell } from 'lucide-react';
import logo from '../assets/logo.png';
import './Dashboard.css';
import { useNavigate } from 'react-router-dom';
import MapComponent from './Map';

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

            // Ease out cubic
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

    return (
        <div className="dashboard-container">
            <header className="dashboard-header">
                <div className="header-left">
                    <img src={logo} alt="Argus Logo" className="header-logo" />
                    <span className="header-title">ARGUS</span>
                </div>
                <div className="header-right">
                    <div className="user-profile">
                        <User size={24} fill="#00ff84" />
                        <span>John Doe</span>
                    </div>
                    <Bell size={24} className="notification-bell" fill="#ff3b3b" />
                </div>
            </header>

            <main className="dashboard-content">
                <div className="stats-grid">
                    <div className="stat-card">
                        <span className="stat-title">Total Cases</span>
                        <span className="stat-value white">
                            <CountUp end={6} />
                        </span>
                    </div>
                    <div className="stat-card">
                        <span className="stat-title">Missing</span>
                        <span className="stat-value red">
                            <CountUp end={2} />
                        </span>
                    </div>
                    <div className="stat-card">
                        <span className="stat-title">Investigating</span>
                        <span className="stat-value yellow">
                            <CountUp end={1} />
                        </span>
                    </div>
                    <div className="stat-card">
                        <span className="stat-title">Found</span>
                        <span className="stat-value green">
                            <CountUp end={2} />
                        </span>
                    </div>
                    <div className="stat-card">
                        <span className="stat-title">Cold Cases</span>
                        <span className="stat-value blue">
                            <CountUp end={1} />
                        </span>
                    </div>
                </div>

                <div className="map-section">
                    <MapComponent />
                </div>
                <div className="action-buttons">
                    <button className="action-btn">Find a Missing Person</button>
                    <button className="action-btn" onClick={() => navigate('/history')}>History</button>
                </div>
            </main>
        </div>
    );
};

export default Dashboard;
