import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Bell, User as UserIcon, PlusCircle, XCircle, Play, CheckCircle, Trash2, MapPin, Navigation, Loader, Brain } from 'lucide-react';
import { db, storage } from '../firebaseConfig';
import { doc, setDoc, serverTimestamp } from 'firebase/firestore';
import { ref, uploadBytesResumable, getDownloadURL } from 'firebase/storage';
import logo from '../assets/logo.png';
import Notifications from './Notifications';
import UserProfileModal from './UserProfileModal';
import { useAuth } from '../hooks/useAuth';
import { getDeviceGPS, reverseGeocode } from '../utils/geoService';
import { sendMediaToModel } from '../utils/embeddingService';
import './ReportCase.css';
import './History.css'; 

const ReportCase = () => {
    const navigate = useNavigate();
    const { currentUser } = useAuth();

    const [formData, setFormData] = useState({
        caseId: '',
        caseType: '',
        name: '',
        nic: '',
        age: '',
        gender: '',
        locationName: '',
        latitude: '',
        longitude: ''
    });

    const [images, setImages] = useState([]);
    const [videos, setVideos] = useState([]);
    const [showSuccessModal, setShowSuccessModal] = useState(false);
    const [showNotifications, setShowNotifications] = useState(false);
    const [showProfile, setShowProfile] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [isDetectingGPS, setIsDetectingGPS] = useState(false);
    const [processingPhase, setProcessingPhase] = useState(null);

    const imageInputRef = useRef(null);
    const videoInputRef = useRef(null);
    const uploadTasksRef = useRef([]);

    const handleDetectGPS = async () => {
        setIsDetectingGPS(true);
        try {
            const coords = await getDeviceGPS();
            const geoData = await reverseGeocode(coords.lat, coords.lng);
            setFormData(prev => ({
                ...prev,
                latitude: coords.lat.toString(),
                longitude: coords.lng.toString(),
                locationName: geoData.displayName || `${geoData.city}, ${geoData.district}`
            }));
        } catch (error) {
            console.error("GPS error:", error);
            alert("Unable to acquire live GPS signal: " + (error.message || "Permission denied or signal unavailable. You can enter coordinates manually."));
        } finally {
            setIsDetectingGPS(false);
        }
    };

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
        if (!formData.caseType || !formData.name || !formData.nic || !formData.age || !formData.gender || !formData.locationName || !formData.latitude || !formData.longitude) {
            alert("Please fill in all mandatory details, including geolocation coordinates (or click Auto-Detect GPS).");
            return;
        }

        const nicRegex = /^([0-9]{9}[vV]|[0-9]{12})$/;
        if (!nicRegex.test(formData.nic)) {
            alert("Invalid NIC format");
            return;
        }

        setIsUploading(true);
        setProcessingPhase('uploading');

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

            const lastSeenLocation = {
                name: formData.locationName,
                lat: parseFloat(formData.latitude),
                lng: parseFloat(formData.longitude),
                source: "Hybrid GPS / Camera Geocoding"
            };

            const victimRef = doc(db, 'victims', caseId);
            await setDoc(victimRef, {
                ...formData,
                caseId: caseId,
                status: 'Investigating',
                lastSeenLocation: lastSeenLocation,
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

            setProcessingPhase('extracting');
            try {
                const modelResult = await sendMediaToModel(caseId, formData.name, images, videos);
                if (!modelResult.success) {
                    console.warn('[ARGUS] Secondary ML model notice:', modelResult.errors);
                } else {
                    console.log('[ARGUS] Media processed successfully.');
                }
            } catch (mlErr) {
                console.warn('[ARGUS] ML model enrollment notice:', mlErr);
            }

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
            setProcessingPhase(null);
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
            gender: '',
            locationName: '',
            latitude: '',
            longitude: ''
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
                        <UserIcon size={22} fill="#d6e4ea" color="#d6e4ea" />
                        <span>{currentUser?.username || 'John Doe'}</span>
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

            <main className="report-content">
                <div className="report-container">
                    <button className="close-btn" onClick={handleClose}>
                        <XCircle size={28} fill="#E53935" color="#ffffff" />
                    </button>

                    <div className="report-container-header">
                        <PlusCircle size={28} color="#5ce1e6" />
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

                            <div className="hybrid-gps-container" style={{ 
                                marginTop: '0.2rem', 
                                padding: '0.85rem', 
                                background: 'rgba(35, 38, 43, 0.65)', 
                                border: '1px solid var(--border)', 
                                borderRadius: '8px',
                                display: 'flex',
                                flexDirection: 'column',
                                gap: '0.65rem'
                            }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                    <span style={{ fontSize: '0.82rem', fontWeight: '700', color: 'var(--ice)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                                        📍 Camera / GPS Geolocation
                                    </span>
                                    <button
                                        type="button"
                                        onClick={handleDetectGPS}
                                        disabled={isDetectingGPS}
                                        style={{
                                            padding: '0.35rem 0.7rem',
                                            background: isDetectingGPS ? 'var(--surface)' : 'var(--grad-primary)',
                                            border: '1px solid var(--border)',
                                            borderRadius: '6px',
                                            color: '#fff',
                                            cursor: isDetectingGPS ? 'not-allowed' : 'pointer',
                                            fontSize: '0.78rem',
                                            fontWeight: '700',
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: '0.35rem',
                                            transition: 'transform 0.2s'
                                        }}
                                    >
                                        {isDetectingGPS ? <Loader size={13} className="spinner" /> : <Navigation size={13} />}
                                        {isDetectingGPS ? "Detecting GPS..." : "Auto-Detect GPS"}
                                    </button>
                                </div>

                                <div className="form-group" style={{ gap: '0.5rem', margin: 0 }}>
                                    <label style={{ minWidth: '70px', fontSize: '0.8rem' }}>Location</label>
                                    <input 
                                        type="text" 
                                        name="locationName" 
                                        placeholder="e.g. Colombo Fort / Camera #4" 
                                        value={formData.locationName} 
                                        onChange={handleInputChange} 
                                        style={{ height: '36px', fontSize: '0.82rem' }}
                                    />
                                </div>

                                <div style={{ display: 'flex', gap: '0.5rem' }}>
                                    <div className="form-group" style={{ gap: '0.5rem', flex: 1, margin: 0 }}>
                                        <label style={{ minWidth: '40px', width: 'auto', fontSize: '0.8rem' }}>Lat</label>
                                        <input 
                                            type="number" 
                                            step="any"
                                            name="latitude" 
                                            placeholder="6.9271" 
                                            value={formData.latitude} 
                                            onChange={handleInputChange}
                                            style={{ height: '36px', fontSize: '0.82rem' }} 
                                        />
                                    </div>
                                    <div className="form-group" style={{ gap: '0.5rem', flex: 1, margin: 0 }}>
                                        <label style={{ minWidth: '40px', width: 'auto', fontSize: '0.8rem' }}>Lng</label>
                                        <input 
                                            type="number" 
                                            step="any"
                                            name="longitude" 
                                            placeholder="79.8612" 
                                            value={formData.longitude} 
                                            onChange={handleInputChange}
                                            style={{ height: '36px', fontSize: '0.82rem' }} 
                                        />
                                    </div>
                                </div>
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
                                    <CheckCircle size={48} color="#5ce1e6" strokeWidth={2} />
                                ) : (
                                    <PlusCircle size={48} color="#a0e4e8" strokeWidth={2} />
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
                                    <CheckCircle size={48} color="#5ce1e6" strokeWidth={2} />
                                ) : (
                                    <PlusCircle size={48} color="#a0e4e8" strokeWidth={2} />
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

                        {processingPhase === 'extracting' ? (
                            <>
                                <h3 style={{ color: 'var(--ice)', marginTop: '1.5rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
                                    <Brain size={22} color="#5ce1e6" /> Extracting Biometric Embeddings...
                                </h3>
                                <p style={{ marginTop: '1rem', color: 'var(--text-secondary)' }}>
                                    Generating ByGaitLight (256D) &amp; OSNet (512D) biometric embeddings.
                                </p>
                                <div style={{ marginTop: '0.75rem', padding: '0.5rem 1rem', background: 'rgba(92,225,230,0.08)', borderRadius: '6px', border: '1px solid rgba(92,225,230,0.2)' }}>
                                    <span style={{ fontSize: '0.78rem', color: '#5ce1e6', fontWeight: '600' }}>🧠 ByGaitLight + OSNet Biometric Enrollment</span>
                                </div>
                            </>
                        ) : (
                            <>
                                <h3 style={{ color: 'var(--ice)', marginTop: '1.5rem' }}>Uploading Data...</h3>
                                <p style={{ marginTop: '1rem', color: 'var(--text-secondary)' }}>Please wait, securing case files...</p>
                                <button className="cancel-upload-btn" onClick={handleCancelUpload}>Cancel Upload</button>
                            </>
                        )}
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
