import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Bell, User as UserIcon, PlusCircle, XCircle, Play, CheckCircle } from 'lucide-react';
import logo from '../assets/logo.png';
import Notifications from './Notifications';
import UserProfileModal from './UserProfileModal';
import './ReportCase.css';
import './History.css'; // Reusing header styles

const ReportCase = () => {
    const navigate = useNavigate();

    const [formData, setFormData] = useState({
        caseId: '',
        caseName: '',
        name: '',
        nic: '',
        age: '',
        gender: ''
    });

    const [images, setImages] = useState([]);
    const [videos, setVideos] = useState([]);
    const [showSuccessModal, setShowSuccessModal] = useState(false);
    const [showNotifications, setShowNotifications] = useState(false);
    const [showProfile, setShowProfile] = useState(false);

    const imageInputRef = useRef(null);
    const videoInputRef = useRef(null);

    const handleInputChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({ ...prev, [name]: value }));
    };

    const handleImageChange = (e) => {
        if (e.target.files) {
            const selectedFiles = Array.from(e.target.files);
            const validImages = selectedFiles.filter(file => file.type.startsWith('image/'));
            
            if (validImages.length !== selectedFiles.length) {
                alert('Please select ONLY image files (e.g., .jpg, .png).');
            }
            setImages(validImages);
        }
    };

    const handleVideoChange = (e) => {
        if (e.target.files) {
            const selectedFiles = Array.from(e.target.files);
            const validVideos = selectedFiles.filter(file => file.type.startsWith('video/'));
            
            if (validVideos.length !== selectedFiles.length) {
                alert('Please select ONLY video files (e.g., .mp4, .mov).');
            }
            setVideos(validVideos);
        }
    };

    const handleBack = () => {
        navigate(-1);
    };

    const handleClose = () => {
        navigate('/dashboard');
    };

    const handleModalClose = () => {
        setShowSuccessModal(false);
        setFormData({
            caseId: '',
            caseName: '',
            name: '',
            nic: '',
            age: '',
            gender: ''
        });
        setImages([]);
        setVideos([]);
    };

    return (
        <div className="report-page">
            <Notifications isOpen={showNotifications} onClose={() => setShowNotifications(false)} />
            <UserProfileModal isOpen={showProfile} onClose={() => setShowProfile(false)} />
            
            <header className="history-header">
                <div className="history-header-left">
                    <button className="history-back-btn" onClick={handleBack}>
                        <ArrowLeft size={24} />
                    </button>
                    <img src={logo} alt="Argus Logo" className="history-logo" />
                    <span className="history-title-text">ARGUS</span>
                </div>
                <div className="history-header-right">
                    <div className="user-profile" onClick={() => setShowProfile(true)} style={{ cursor: 'pointer' }}>
                        <UserIcon size={24} fill="#00ff84" />
                        <span>John Doe</span>
                    </div>
                    <Bell 
                        size={24} 
                        className="notification-bell" 
                        fill="#ff3b3b" 
                        onClick={() => setShowNotifications(true)}
                        style={{ cursor: 'pointer' }}
                    />
                </div>
            </header>

            <main className="report-content">
                <div className="report-container">
                    <button className="close-btn" onClick={handleClose}>
                        <XCircle size={32} fill="#ef4444" color="#1a1c29" />
                    </button>

                    <div className="report-container-header">
                        <PlusCircle size={28} color="#00ff84" />
                        <h2>Report a New Case</h2>
                    </div>

                    <div className="report-form-layout">
                        <div className="report-form-left">
                            <div className="form-group">
                                <label>Case ID</label>
                                <input type="text" name="caseId" value={formData.caseId} onChange={handleInputChange} />
                            </div>
                            <div className="form-group">
                                <label>Case Name</label>
                                <input type="text" name="caseName" value={formData.caseName} onChange={handleInputChange} />
                            </div>
                            <div className="form-group">
                                <label>Name</label>
                                <input type="text" name="name" value={formData.name} onChange={handleInputChange} />
                            </div>
                            <div className="form-group">
                                <label>NIC</label>
                                <input type="text" name="nic" value={formData.nic} onChange={handleInputChange} />
                            </div>
                            <div className="form-group">
                                <label>Age</label>
                                <input type="number" name="age" value={formData.age} onChange={handleInputChange} />
                            </div>
                            <div className="form-group">
                                <label>Gender</label>
                                <input type="text" name="gender" value={formData.gender} onChange={handleInputChange} />
                            </div>
                        </div>

                        <div className="report-form-right">
                            <input 
                                type="file" 
                                multiple 
                                accept="image/*" 
                                style={{ display: 'none' }} 
                                ref={imageInputRef} 
                                onChange={handleImageChange} 
                            />
                            <div 
                                className={`upload-box ${images.length > 0 ? 'has-files' : ''}`} 
                                onClick={() => imageInputRef.current.click()}
                            >
                                {images.length > 0 ? (
                                    <CheckCircle size={48} color="#00ff84" strokeWidth={2} />
                                ) : (
                                    <PlusCircle size={48} color="#000" strokeWidth={2} />
                                )}
                                <span>{images.length > 0 ? `${images.length} Image(s) Added` : 'Add Images'}</span>
                            </div>

                            <input 
                                type="file" 
                                multiple 
                                accept="video/*" 
                                style={{ display: 'none' }} 
                                ref={videoInputRef} 
                                onChange={handleVideoChange} 
                            />
                            <div 
                                className={`upload-box ${videos.length > 0 ? 'has-files' : ''}`} 
                                onClick={() => videoInputRef.current.click()}
                            >
                                {videos.length > 0 ? (
                                    <CheckCircle size={48} color="#00ff84" strokeWidth={2} />
                                ) : (
                                    <PlusCircle size={48} color="#000" strokeWidth={2} />
                                )}
                                <span>{videos.length > 0 ? `${videos.length} Video(s) Added` : 'Add Videos'}</span>
                            </div>

                            <button className="submit-btn" onClick={() => setShowSuccessModal(true)}>
                                <Play size={32} fill="#000" color="#000" />
                            </button>
                        </div>
                    </div>
                </div>
            </main>

            {showSuccessModal && (
                <div className="modal-overlay">
                    <div className="success-modal">
                        <button className="modal-close-btn" onClick={handleModalClose}>
                            <XCircle size={32} fill="#ef4444" color="#484848" />
                        </button>
                        <h3>New Case added ( Case id : {formData.caseId || '______'} )</h3>
                    </div>
                </div>
            )}
        </div>
    );
};

export default ReportCase;
