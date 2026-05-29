import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Bell, User as UserIcon, PlusCircle, XCircle, Play, CheckCircle, Trash2 } from 'lucide-react';
import { db, storage } from '../firebaseConfig';
import { doc, setDoc, serverTimestamp } from 'firebase/firestore';
import { ref, uploadBytesResumable, getDownloadURL } from 'firebase/storage';
import logo from '../assets/logo.png';
import Notifications from './Notifications';
import UserProfileModal from './UserProfileModal';
import './ReportCase.css';
import './History.css'; 

const ReportCase = () => {
    const navigate = useNavigate();

    const [formData, setFormData] = useState({
        caseId: '',
        caseType: '',
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
    const [isUploading, setIsUploading] = useState(false);

    const imageInputRef = useRef(null);
    const videoInputRef = useRef(null);
    const uploadTasksRef = useRef([]);

    const handleCancelUpload = () => {
        uploadTasksRef.current.forEach(task => task.cancel());
        uploadTasksRef.current = [];
        setIsUploading(false);
    };

    const handleInputChange = (e) => {
        let { name, value } = e.target;
        setFormData(prev => {
            const updated = { ...prev };
            if (name === 'nic') {
                value = value.replace(/[^0-9vV]/g, '');
                updated[name] = value;
                const last4 = value.length >= 4 ? value.slice(-4) : value;
                updated.caseId = `Case-${last4}`;
            } else {
                updated[name] = value;
            }
            return updated;
        });
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

    const handleRemoveImage = (index) => {
        setImages((prev) => prev.filter((_, idx) => idx !== index));
    };

    const handleBack = () => navigate(-1);
    const handleClose = () => navigate('/dashboard');

    const handleSubmit = async () => {
        if (!formData.caseType || !formData.name || !formData.nic || !formData.age || !formData.gender) {
            alert("Please fill in all mandatory details.");
            return;
        }

        const nicRegex = /^([0-9]{9}[vV]|[0-9]{12})$/;
        if (!nicRegex.test(formData.nic)) {
            alert("Invalid NIC format");
            return;
        }

        setIsUploading(true);

        try {
            const caseId = formData.caseId;
            console.log('Submitting case', caseId, 'NIC', formData.nic);

            
            const imageUrls = [];
            for (let i = 0; i < images.length; i++) {
                const file = images[i];
                const fileRef = ref(storage, `cases/${caseId}/images/${file.name}`);
                const uploadTask = uploadBytesResumable(fileRef, file);
                uploadTasksRef.current.push(uploadTask);
                await new Promise((resolve, reject) => {
                    uploadTask.on('state_changed', null, reject, () => resolve());
                });
                const url = await getDownloadURL(fileRef);
                imageUrls.push(url);
            }

            
            const videoUrls = [];
            for (let i = 0; i < videos.length; i++) {
                const file = videos[i];
                const fileRef = ref(storage, `cases/${caseId}/videos/${file.name}`);
                const uploadTask = uploadBytesResumable(fileRef, file);
                uploadTasksRef.current.push(uploadTask);
                await new Promise((resolve, reject) => {
                    uploadTask.on('state_changed', null, reject, () => resolve());
                });
                const url = await getDownloadURL(fileRef);
                videoUrls.push(url);
            }

            
            const victimRef = doc(db, 'victims', caseId);
            await setDoc(victimRef, {
                ...formData,
                caseId: caseId,
                status: 'Active',
                createdAt: serverTimestamp()
            });

            
            const mediaRef = doc(db, 'person_media', caseId);
            await setDoc(mediaRef, {
                caseId: caseId,
                nic: formData.nic,
                imageUrls: imageUrls,
                videoUrls: videoUrls,
                linkedAt: serverTimestamp(),
                createdAt: serverTimestamp()
            });

            
            setShowSuccessModal(true);
        } catch (error) {
            if (error.code === 'storage/canceled') {
                console.log("Upload was cancelled by the user.");
            } else {
                console.error("Error uploading case:", error);
                alert("Firebase Error: " + error.message + "\n\nPlease check your Firebase Security Rules or ensure you selected valid files.");
            }
        } finally {
            setIsUploading(false);
            uploadTasksRef.current = [];
        }
    };

    const handleModalClose = () => {
        setShowSuccessModal(false);
        setFormData({
            caseId: '',
            caseType: '',
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
                        <UserIcon size={22} fill="#CAF0F8" color="#CAF0F8" />
                        <span>John Doe</span>
                    </div>
                    <Bell
                        size={22}
                        className="notification-bell"
                        fill="#00B4D8"
                        color="#00B4D8"
                        onClick={() => setShowNotifications(true)}
                        style={{ cursor: 'pointer' }}
                    />
                </div>
            </header>

            <main className="report-content">
                <div className="report-container">
                    <button className="close-btn" onClick={handleClose}>
                        <XCircle size={28} fill="#E53935" color="#ffffff" />
                    </button>

                    <div className="report-container-header">
                        <PlusCircle size={28} color="#00B4D8" />
                        <h2>Report a New Case</h2>
                    </div>

                    <div className="report-form-layout">
                        <div className="report-form-left">
                            <div className="form-group">
                                <label>Case ID</label>
                                <input type="text" name="caseId" value={formData.caseId} readOnly />
                            </div>
                            <div className="form-group">
                                <label>Case Type</label>
                                <select name="caseType" value={formData.caseType} onChange={handleInputChange}>
                                    <option value="" disabled>Select Case Type</option>
                                    <option value="Missing">Missing</option>
                                    <option value="Kidnapping">Kidnapping</option>
                                    <option value="Abduction">Abduction</option>
                                    <option value="Robbery">Robbery</option>
                                </select>
                            </div>
                            <div className="form-group">
                                <label>Name</label>
                                <input type="text" name="name" value={formData.name} onChange={handleInputChange} />
                            </div>
                            <div className="form-group">
                                <label>NIC</label>
                                <input type="text" name="nic" value={formData.nic} onChange={handleInputChange} maxLength="12" />
                            </div>
                            <div className="form-group">
                                <label>Age</label>
                                <input type="number" name="age" value={formData.age} onChange={handleInputChange} />
                            </div>
                            <div className="form-group">
                                <label>Gender</label>
                                <select name="gender" value={formData.gender} onChange={handleInputChange}>
                                    <option value="" disabled>Select Gender</option>
                                    <option value="Male">Male</option>
                                    <option value="Female">Female</option>
                                    <option value="Prefer not to say">Prefer not to say</option>
                                </select>
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
                                    <CheckCircle size={48} color="#00B4D8" strokeWidth={2} />
                                ) : (
                                    <PlusCircle size={48} color="#90E0EF" strokeWidth={2} />
                                )}
                                <span>{images.length > 0 ? `${images.length} Image(s) Added` : 'Add Images'}</span>
                            </div>

                            {images.length > 0 && (
                                <div className="selected-images">
                                    <div className="selected-images-header">Selected Images</div>
                                    {images.map((image, index) => (
                                        <div key={image.name + index} className="selected-image-item">
                                            <span className="selected-image-name">{image.name}</span>
                                            <button
                                                type="button"
                                                className="selected-image-remove"
                                                onClick={() => handleRemoveImage(index)}
                                                aria-label={`Remove ${image.name}`}
                                            >
                                                <Trash2 size={16} />
                                            </button>
                                        </div>
                                    ))}
                                </div>
                            )}

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
                                    <CheckCircle size={48} color="#00B4D8" strokeWidth={2} />
                                ) : (
                                    <PlusCircle size={48} color="#90E0EF" strokeWidth={2} />
                                )}
                                <span>{videos.length > 0 ? `${videos.length} Video(s) Added` : 'Add Videos'}</span>
                            </div>

                            <button className="submit-btn" onClick={handleSubmit} disabled={isUploading}>
                                <Play size={32} fill={isUploading ? "#555" : "#fff"} color={isUploading ? "#555" : "#fff"} />
                            </button>
                        </div>
                    </div>
                </div>
            </main>

            {isUploading && (
                <div className="modal-overlay">
                    <div className="success-modal" style={{ textAlign: 'center' }}>
                        <div className="spinner"></div>
                        <h3 style={{ color: 'var(--ice)', marginTop: '1.5rem' }}>Uploading Data...</h3>
                        <p style={{ marginTop: '1rem', color: 'var(--text-secondary)' }}>Please wait, securing case files...</p>
                        <button className="cancel-upload-btn" onClick={handleCancelUpload}>Cancel Upload</button>
                    </div>
                </div>
            )}

            {showSuccessModal && (
                <div className="modal-overlay" onClick={handleModalClose}>
                    <div className="success-modal" onClick={(e) => e.stopPropagation()}>
                        <button className="modal-close-btn" onClick={handleModalClose}>
                            <XCircle size={28} fill="#E53935" color="#ffffff" />
                        </button>
                        <h3>New Case added ( Case id : {formData.caseId} )</h3>
                    </div>
                </div>
            )}
        </div>
    );
};

export default ReportCase;
