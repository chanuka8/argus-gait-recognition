import React, { useState, useEffect, useRef } from 'react';
import { Video, VideoOff, Plus, Trash2, ShieldAlert, Cpu, Radio, MapPin } from 'lucide-react';
import AdminHeader from './AdminHeader';
import './SurveillanceFeeds.css';

// Component to render a dynamic canvas-based "live" terminal camera stream
const LiveCameraFeed = ({ isOnline, cameraName }) => {
    const canvasRef = useRef(null);

    useEffect(() => {
        if (!isOnline) return;

        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        let animationFrameId;

        // Set high dimensions for clarity
        canvas.width = 300;
        canvas.height = 180;

        let frameCount = 0;
        const points = Array.from({ length: 8 }, () => ({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 2,
            vy: (Math.random() - 0.5) * 2
        }));

        const draw = () => {
            // Clear background
            ctx.fillStyle = '#02031a';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Draw grid lines
            ctx.strokeStyle = 'rgba(92, 225, 230, 0.08)';
            ctx.lineWidth = 1;
            const gridSize = 20;
            for (let x = 0; x < canvas.width; x += gridSize) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, canvas.height);
                ctx.stroke();
            }
            for (let y = 0; y < canvas.height; y += gridSize) {
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(canvas.width, y);
                ctx.stroke();
            }

            // Draw abstract moving nodes (simulating radar detection objects)
            ctx.fillStyle = 'rgba(144, 224, 239, 0.5)';
            ctx.strokeStyle = 'rgba(92, 225, 230, 0.3)';
            points.forEach(p => {
                p.x += p.vx;
                p.y += p.vy;

                // Bounce off walls
                if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
                if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

                ctx.beginPath();
                ctx.arc(p.x, p.y, 3, 0, Math.PI * 2);
                ctx.fill();

                // Draw radar signal range circle around the first node
                ctx.beginPath();
                ctx.arc(p.x, p.y, 25, 0, Math.PI * 2);
                ctx.stroke();
            });

            // Draw connections between close nodes
            ctx.strokeStyle = 'rgba(92, 225, 230, 0.15)';
            for (let i = 0; i < points.length; i++) {
                for (let j = i + 1; j < points.length; j++) {
                    const dist = Math.hypot(points[i].x - points[j].x, points[i].y - points[j].y);
                    if (dist < 60) {
                        ctx.beginPath();
                        ctx.moveTo(points[i].x, points[i].y);
                        ctx.lineTo(points[j].x, points[j].y);
                        ctx.stroke();
                    }
                }
            }

            // Static/noise overlay (simulating security camera feed grain)
            ctx.fillStyle = 'rgba(255, 255, 255, 0.04)';
            for (let i = 0; i < 400; i++) {
                const rx = Math.random() * canvas.width;
                const ry = Math.random() * canvas.height;
                ctx.fillRect(rx, ry, 1, 1);
            }

            // Scanner Sweep line
            const scannerY = (frameCount * 2) % canvas.height;
            ctx.strokeStyle = 'rgba(0, 230, 118, 0.2)';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(0, scannerY);
            ctx.lineTo(canvas.width, scannerY);
            ctx.stroke();

            // Dynamic timestamp overlay
            ctx.fillStyle = 'rgba(202, 240, 248, 0.8)';
            ctx.font = '9px Courier New';
            ctx.fillText(`CAM_${cameraName.toUpperCase().replace(/\s+/g, '_')}`, 10, 18);
            ctx.fillText(`REC: ${new Date().toISOString().slice(0, 19).replace('T', ' ')}`, 10, 30);
            ctx.fillText('SECURE_FEED_LINK_OK', 10, 42);

            frameCount++;
            animationFrameId = requestAnimationFrame(draw);
        };

        draw();

        return () => {
            cancelAnimationFrame(animationFrameId);
        };
    }, [isOnline, cameraName]);

    if (!isOnline) {
        return (
            <div className="camera-offline-screen">
                <VideoOff size={40} color="var(--status-missing)" />
                <p>FEED TERMINATED</p>
                <span>No active connection handshake</span>
            </div>
        );
    }

    return <canvas ref={canvasRef} className="camera-feed-canvas" />;
};

const SurveillanceFeeds = () => {
    const defaultCameras = [
        { id: 1, name: 'Main Lobby Entrance', sector: 'Sector Alpha', fps: 30, bitrate: 2100, isOnline: true, location: 'Latitude: 7.2905, Longitude: 80.6337' },
        { id: 2, name: 'North Boundary Perimeter', sector: 'Sector Beta', fps: 24, bitrate: 1400, isOnline: true, location: 'Latitude: 7.2912, Longitude: 80.6345' },
        { id: 3, name: 'Core Server Vault', sector: 'Sector Delta', fps: 60, bitrate: 4500, isOnline: true, location: 'Latitude: 7.2898, Longitude: 80.6322' },
        { id: 4, name: 'South Transit Bay', sector: 'Sector Gamma', fps: 30, bitrate: 1800, isOnline: false, location: 'Latitude: 7.2918, Longitude: 80.6310' }
    ];

    const [cameras, setCameras] = useState(() => {
        const localCams = localStorage.getItem('argus_surveillance_cameras');
        return localCams ? JSON.parse(localCams) : defaultCameras;
    });

    const [newCamName, setNewCamName] = useState('');
    const [newCamSector, setNewCamSector] = useState('Sector Alpha');
    const [newCamLocation, setNewCamLocation] = useState('Latitude: 7.2900, Longitude: 80.6300');
    const [newCamFps, setNewCamFps] = useState(30);
    const [newCamBitrate, setNewCamBitrate] = useState(2000);
    const [showAddForm, setShowAddForm] = useState(false);

    useEffect(() => {
        localStorage.setItem('argus_surveillance_cameras', JSON.stringify(cameras));
    }, [cameras]);

    const handleAddCamera = (e) => {
        e.preventDefault();
        if (!newCamName.trim()) return;

        const newCamera = {
            id: Date.now(),
            name: newCamName,
            sector: newCamSector,
            fps: parseInt(newCamFps),
            bitrate: parseInt(newCamBitrate),
            isOnline: true,
            location: newCamLocation
        };

        setCameras([...cameras, newCamera]);
        setNewCamName('');
        setShowAddForm(false);
    };

    const toggleCameraStatus = (id) => {
        setCameras(cameras.map(cam => 
            cam.id === id ? { ...cam, isOnline: !cam.isOnline } : cam
        ));
    };

    const deleteCamera = (id) => {
        if (window.confirm('Remove this camera feed connection?')) {
            setCameras(cameras.filter(cam => cam.id !== id));
        }
    };

    return (
        <div className="surveillance-feeds-page">
            <AdminHeader />

            <main className="surveillance-content">
                <div className="surv-header-row">
                    <div className="title-group">
                        <h1>Surveillance Feed Administration</h1>
                        <p>Configure live stream endpoints, monitor packet telemetry, and verify perimeter sensor grids.</p>
                    </div>
                    <button className="add-feed-btn" onClick={() => setShowAddForm(!showAddForm)}>
                        <Plus size={18} />
                        <span>{showAddForm ? 'Cancel Registration' : 'Register Stream'}</span>
                    </button>
                </div>

                {showAddForm && (
                    <form onSubmit={handleAddCamera} className="add-camera-form-panel">
                        <h3>New Camera Connection</h3>
                        <div className="form-grid">
                            <div className="form-field">
                                <label>Camera Name</label>
                                <input 
                                    type="text" 
                                    placeholder="e.g. CCTV 05 - Loading Dock" 
                                    value={newCamName} 
                                    onChange={(e) => setNewCamName(e.target.value)} 
                                    required 
                                />
                            </div>
                            <div className="form-field">
                                <label>Operating Sector</label>
                                <select value={newCamSector} onChange={(e) => setNewCamSector(e.target.value)}>
                                    <option value="Sector Alpha">Sector Alpha</option>
                                    <option value="Sector Beta">Sector Beta</option>
                                    <option value="Sector Gamma">Sector Gamma</option>
                                    <option value="Sector Delta">Sector Delta</option>
                                </select>
                            </div>
                            <div className="form-field">
                                <label>Vector Coordinates</label>
                                <input 
                                    type="text" 
                                    value={newCamLocation} 
                                    onChange={(e) => setNewCamLocation(e.target.value)} 
                                    placeholder="Latitude / Longitude"
                                    required 
                                />
                            </div>
                            <div className="form-field">
                                <label>Stream framerate (FPS)</label>
                                <input 
                                    type="number" 
                                    value={newCamFps} 
                                    onChange={(e) => setNewCamFps(e.target.value)} 
                                    min="10" 
                                    max="60" 
                                    required 
                                />
                            </div>
                            <div className="form-field">
                                <label>Target Bitrate (kbps)</label>
                                <input 
                                    type="number" 
                                    value={newCamBitrate} 
                                    onChange={(e) => setNewCamBitrate(e.target.value)} 
                                    min="500" 
                                    max="10000" 
                                    required 
                                />
                            </div>
                        </div>
                        <button type="submit" className="submit-cam-btn">Authorize Terminal Connection</button>
                    </form>
                )}

                <div className="camera-feeds-layout-grid">
                    {cameras.map((cam) => (
                        <div key={cam.id} className={`camera-feed-card ${cam.isOnline ? 'online' : 'offline'}`}>
                            <div className="camera-feed-viewport">
                                <LiveCameraFeed isOnline={cam.isOnline} cameraName={cam.name} />
                                <div className="cam-viewport-overlay">
                                    <span className="cam-fps">{cam.isOnline ? `${cam.fps} FPS` : '0 FPS'}</span>
                                    <span className="cam-bitrate">{cam.isOnline ? `${cam.bitrate} kbps` : '0 kbps'}</span>
                                </div>
                            </div>

                            <div className="camera-details-body">
                                <div className="camera-title-row">
                                    <h3>{cam.name}</h3>
                                    <span className={`cam-status-dot-badge ${cam.isOnline ? 'online' : 'offline'}`}>
                                        {cam.isOnline ? 'LIVE' : 'OFFLINE'}
                                    </span>
                                </div>
                                
                                <div className="camera-meta-fields">
                                    <div className="meta-field">
                                        <Radio size={14} />
                                        <span>{cam.sector}</span>
                                    </div>
                                    <div className="meta-field">
                                        <MapPin size={14} />
                                        <span className="meta-loc">{cam.location}</span>
                                    </div>
                                </div>

                                <div className="camera-card-actions">
                                    <button 
                                        className={`toggle-connection-btn ${cam.isOnline ? 'terminate' : 'initiate'}`} 
                                        onClick={() => toggleCameraStatus(cam.id)}
                                    >
                                        {cam.isOnline ? 'Terminate Link' : 'Establish Link'}
                                    </button>
                                    <button className="remove-camera-btn" onClick={() => deleteCamera(cam.id)}>
                                        <Trash2 size={16} />
                                    </button>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </main>
        </div>
    );
};

export default SurveillanceFeeds;
