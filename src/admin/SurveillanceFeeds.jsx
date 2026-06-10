import React from 'react';
import { VideoOff } from 'lucide-react';
import AdminHeader from './AdminHeader';
import './SurveillanceFeeds.css';

const SurveillanceFeeds = () => {
    return (
        <div className="surveillance-feeds-page">
            <AdminHeader />

            <main className="surveillance-content">
                <div className="surv-header-row">
                    <div className="title-group">
                        <h1>CCTV Camera Feeds</h1>
                    </div>
                </div>

                <div className="surveillance-empty-state">
                    <VideoOff size={56} color="var(--status-missing)" />
                    <h2>Cameras Not Connected</h2>
                    <p>Camera access has not been configured yet. This section will be available once surveillance feeds are integrated.</p>
                </div>
            </main>
        </div>
    );
};

export default SurveillanceFeeds;
