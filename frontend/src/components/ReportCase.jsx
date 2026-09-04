import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Bell, User as UserIcon, PlusCircle, XCircle, Play, CheckCircle, Trash2, MapPin, Navigation, Loader, Brain, AlertTriangle } from 'lucide-react';
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

const VIDEO_STAGES = [
    { key: 'QUEUED', label: 'Queued' },
    { key: 'VALIDATING_VIDEO', label: 'Validating' },
    { key: 'TRACKING', label: 'Tracking' },
    { key: 'FEATURE_EXTRACTION', label: 'Feature Extraction' },
    { key: 'MATCHING', label: 'Matching' },
    { key: 'PERSISTING', label: 'Persisting' },
    { key: 'COMPLETED', label: 'Completed' },
];

const formatBytes = (bytes) => {
    if (!bytes || bytes <= 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
};

const getEffectiveStageIndex = (stage, status) => {
    if (status === 'COMPLETED') return 6;
    if (status === 'RESUMING') {
        const normalized = (stage || '').toUpperCase();
        if (normalized === 'FEATURE_EXTRACTION' || normalized === 'GENERATING_GEI' || normalized === 'EXTRACTING_EMBEDDINGS') return 3;
        if (normalized === 'MATCHING') return 4;
        if (normalized === 'PERSISTING') return 5;
        return 2;
    }
    if (!stage) return status === 'QUEUED' ? 0 : 0;
    const normalized = stage.toUpperCase();
    if (normalized === 'QUEUED') return 0;
    if (normalized === 'VALIDATING_VIDEO') return 1;
    if (normalized === 'TRACKING') return 2;
    if (normalized === 'FEATURE_EXTRACTION' || normalized === 'GENERATING_GEI' || normalized === 'EXTRACTING_EMBEDDINGS') return 3;
    if (normalized === 'MATCHING') return 4;
    if (normalized === 'PERSISTING') return 5;
    if (normalized === 'COMPLETED') return 6;
    return 2;
};

const calculateVideoProgressPercent = (stage, status, framesProcessed, totalFrames) => {
    if (status === 'COMPLETED') return 100;
    if (status === 'RESUMING') {
        if (totalFrames && totalFrames > 0 && framesProcessed > 0) {
            return Math.min(95, Math.max(25, Math.round((framesProcessed / totalFrames) * 75)));
        }
        return 35;
    }
    if (!stage) return status === 'QUEUED' ? 5 : 10;
    const normalized = stage.toUpperCase();
    if (normalized === 'QUEUED') return 5;
    if (normalized === 'VALIDATING_VIDEO') return 15;
    if (normalized === 'TRACKING') {
        if (totalFrames && totalFrames > 0) {
            return Math.min(65, 20 + Math.round((framesProcessed / totalFrames) * 45));
        }
        return 40;
    }
    if (normalized === 'FEATURE_EXTRACTION' || normalized === 'GENERATING_GEI' || normalized === 'EXTRACTING_EMBEDDINGS') return 75;
    if (normalized === 'MATCHING') return 88;
    if (normalized === 'PERSISTING') return 95;
    if (normalized === 'COMPLETED') return 100;
    return 50;
};

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
    const [gaitProgress, setGaitProgress] = useState(null);
    const [uploadProgress, setUploadProgress] = useState({
        phase: 'IDLE',
        percent: 0,
        loaded: 0,
        total: 0,
        label: '',
        mediaType: null,
    });
    const [processingProgress, setProcessingProgress] = useState({
        stage: null,
        status: null,
        percent: 0,
        framesProcessed: 0,
        totalFrames: 0,
        fps: 0,
        validSilhouettes: 0,
        validSequences: 0,
        embeddingsCommitted: 0,
    });
    const [enrollmentResult, setEnrollmentResult] = useState(null);
    const [enrollmentWarning, setEnrollmentWarning] = useState(null);
    const [cloudSyncWarning, setCloudSyncWarning] = useState(null);

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

            let imageUrls = [];
            let videoUrls = [];

            // 1. Cloud storage sync (resilient to offline or network drops, never blocks local ARGUS pipeline)
            try {
                const totalFirebaseBytes = [...images, ...videos].reduce((sum, f) => sum + (f.size || 0), 0);
                const transferredMap = {};

                const updateFirebaseProgress = (fileKey, bytesTransferred) => {
                    transferredMap[fileKey] = bytesTransferred;
                    const currentTotal = Object.values(transferredMap).reduce((a, b) => a + b, 0);
                    const pct = totalFirebaseBytes > 0
                        ? Math.min(100, Math.round((currentTotal / totalFirebaseBytes) * 100))
                        : 100;
                    setUploadProgress({
                        phase: 'FIREBASE',
                        percent: pct,
                        loaded: currentTotal,
                        total: totalFirebaseBytes,
                        label: `Syncing files with cloud storage (${pct}%)...`,
                        mediaType: null,
                    });
                };

                setUploadProgress({
                    phase: 'FIREBASE',
                    percent: 0,
                    loaded: 0,
                    total: totalFirebaseBytes,
                    label: 'Syncing files with cloud storage...',
                    mediaType: null,
                });

                for (let i = 0; i < images.length; i++) {
                    const file = images[i];
                    const fileKey = `img_${i}_${file.name}`;
                    const fileRef = ref(storage, `cases/${caseId}/images/${file.name}`);
                    const uploadTask = uploadBytesResumable(fileRef, file);
                    uploadTasksRef.current.push(uploadTask);
                    await new Promise((resolve, reject) => {
                        uploadTask.on(
                            'state_changed',
                            (snapshot) => updateFirebaseProgress(fileKey, snapshot.bytesTransferred),
                            reject,
                            () => {
                                updateFirebaseProgress(fileKey, file.size || 0);
                                resolve();
                            }
                        );
                    });
                    const url = await getDownloadURL(fileRef);
                    imageUrls.push(url);
                }

                for (let i = 0; i < videos.length; i++) {
                    const file = videos[i];
                    const fileKey = `vid_${i}_${file.name}`;
                    const fileRef = ref(storage, `cases/${caseId}/videos/${file.name}`);
                    const uploadTask = uploadBytesResumable(fileRef, file);
                    uploadTasksRef.current.push(uploadTask);
                    await new Promise((resolve, reject) => {
                        uploadTask.on(
                            'state_changed',
                            (snapshot) => updateFirebaseProgress(fileKey, snapshot.bytesTransferred),
                            reject,
                            () => {
                                updateFirebaseProgress(fileKey, file.size || 0);
                                resolve();
                            }
                        );
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
            } catch (cloudErr) {
                if (cloudErr.code === 'storage/canceled') {
                    console.log("Cloud upload cancelled by user.");
                    return;
                }
                console.warn("[ARGUS] Cloud storage sync notice (proceeding with local ARGUS AI ingestion):", cloudErr);
                setCloudSyncWarning("Operating in local-first mode. Biometrics and case data are enrolled and secured in local ARGUS surveillance storage.");
            }

            if (images.length > 0 || videos.length > 0) {
                setUploadProgress({
                    phase: 'BIOMETRIC_UPLOAD',
                    percent: 0,
                    loaded: 0,
                    total: 0,
                    label: videos.length > 0
                        ? 'Uploading reference video to ARGUS AI...'
                        : 'Uploading photos to ARGUS AI...',
                    mediaType: videos.length > 0 ? 'video' : 'image',
                });

                try {
                    const modelResult = await sendMediaToModel(
                        caseId,
                        formData.name,
                        images,
                        videos,
                        (progressData, status, jobData) => {
                            if (progressData.phase === 'UPLOAD') {
                                setUploadProgress({
                                    phase: 'BIOMETRIC_UPLOAD',
                                    percent: progressData.percent || 0,
                                    loaded: progressData.loaded || 0,
                                    total: progressData.total || 0,
                                    speedMBs: progressData.speedMBs || null,
                                    etaSeconds: progressData.etaSeconds || null,
                                    connectionStatus: progressData.connectionStatus || 'stable',
                                    label: progressData.mediaType === 'video'
                                        ? `Uploading reference video to ARGUS AI (${progressData.percent || 0}%)...`
                                        : `Uploading reference photo(s) to ARGUS AI (${progressData.percent || 0}%)...`,
                                    mediaType: progressData.mediaType,
                                });
                            } else if (progressData.phase === 'UPLOAD_COMPLETE') {
                                setUploadProgress(prev => ({
                                    ...prev,
                                    phase: 'UPLOAD_COMPLETE',
                                    percent: 100,
                                    loaded: prev.total || prev.loaded,
                                    speedMBs: '0.00',
                                    etaSeconds: 0,
                                    connectionStatus: 'stable',
                                    label: 'Upload complete. Biometric processing dispatched asynchronously.',
                                }));
                            } else if (progressData.phase === 'PROCESSING' || status) {
                                setUploadProgress(prev => ({
                                    ...prev,
                                    phase: 'UPLOAD_COMPLETE',
                                    percent: 100,
                                    label: 'Upload complete.',
                                }));
                                setProcessingPhase('extracting');
                                const framesProc = progressData.frames_processed || 0;
                                const lastSafe = progressData.last_safe_frame || framesProc;
                                const totFrames = progressData.total_frames || 0;
                                const stageName = progressData.stage || (status === 'QUEUED' ? 'QUEUED' : 'PROCESSING');
                                const serverPct = progressData.percent || 0;
                                const calcPct = progressData.mediaType === 'image'
                                    ? 100
                                    : Math.max(serverPct, calculateVideoProgressPercent(stageName, status, framesProc, totFrames));

                                setProcessingProgress({
                                    stage: stageName,
                                    status: status || 'PROCESSING',
                                    percent: calcPct,
                                    framesProcessed: framesProc,
                                    lastSafeFrame: lastSafe,
                                    totalFrames: totFrames,
                                    fps: progressData.fps || 0,
                                    validSilhouettes: progressData.valid_silhouettes || 0,
                                    validSequences: progressData.valid_sequences || 0,
                                    embeddingsCommitted: progressData.embeddings_committed || progressData.embeddings_generated || 0,
                                    recoveryCount: jobData?.recovery_count || 0,
                                    resumed: jobData?.resumed || false,
                                });
                                setGaitProgress({ ...progressData, status, jobData });
                            }
                        }
                    );
                    if (!modelResult.success || (modelResult.errors && modelResult.errors.length > 0)) {
                        console.warn('[ARGUS] Biometric enrollment notice/warning:', modelResult.errors);
                        setEnrollmentWarning(modelResult.errors.join(' | '));
                    } else {
                        console.log('[ARGUS] Media processed successfully:', modelResult);
                        setEnrollmentResult(modelResult);
                    }
                } catch (mlErr) {
                    console.warn('[ARGUS] ML model enrollment notice:', mlErr);
                    setEnrollmentWarning(mlErr.message || 'Biometric reference processing could not complete.');
                }
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
        setCloudSyncWarning(null);
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
        setUploadProgress({
            phase: 'IDLE',
            percent: 0,
            loaded: 0,
            total: 0,
            label: '',
            mediaType: null,
        });
        setProcessingProgress({
            stage: null,
            status: null,
            percent: 0,
            framesProcessed: 0,
            totalFrames: 0,
            fps: 0,
            validSilhouettes: 0,
            validSequences: 0,
            embeddingsCommitted: 0,
        });
        setGaitProgress(null);
        setEnrollmentResult(null);
        setEnrollmentWarning(null);
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
                    <div className="success-modal" style={{ textAlign: 'center', maxWidth: '480px' }}>
                        <div className="spinner"></div>

                        {processingPhase === 'extracting' ? (
                            <>
                                <div style={{ marginTop: '1rem' }}>
                                    <span className="upload-complete-badge">✓ Media Upload Completed</span>
                                </div>
                                <h3 style={{ color: 'var(--ice)', marginTop: '0.85rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', fontSize: '1.2rem' }}>
                                    <Brain size={22} color="#5ce1e6" /> Extracting Reference Biometrics...
                                </h3>

                                <div className="progress-bar-container">
                                    <div
                                        className="progress-bar-fill processing"
                                        style={{ width: `${processingProgress.percent}%` }}
                                    ></div>
                                </div>
                                <div className="progress-stats">
                                    <span>Processing: {processingProgress.percent}%</span>
                                    <span>Status: {processingProgress.status || 'PROCESSING'}</span>
                                </div>

                                {videos.length > 0 && (
                                    <div className="stage-badge-container">
                                        {VIDEO_STAGES.map((s, idx) => {
                                            const currentIdx = getEffectiveStageIndex(processingProgress.stage, processingProgress.status);
                                            let badgeClass = 'stage-badge';
                                            if (idx < currentIdx) badgeClass += ' done';
                                            else if (idx === currentIdx) badgeClass += ' active';
                                            return (
                                                <span key={s.key} className={badgeClass}>
                                                    {idx < currentIdx ? '✓ ' : ''}{s.label}
                                                </span>
                                            );
                                        })}
                                    </div>
                                )}

                                <p style={{ marginTop: '0.75rem', color: 'var(--text-secondary)', fontSize: '0.84rem' }}>
                                    {processingProgress.status === 'RESUMING'
                                        ? `Resuming previous processing from frame ${processingProgress.lastSafeFrame || processingProgress.framesProcessed || 0} / ${processingProgress.totalFrames || 0}...`
                                        : processingProgress.stage === 'TRACKING'
                                        ? `Tracking Target Subject (${processingProgress.framesProcessed} frames @ ${processingProgress.fps || 0} FPS)...`
                                        : processingProgress.stage === 'GENERATING_GEI' || processingProgress.stage === 'FEATURE_EXTRACTION'
                                        ? `Constructing Gait Energy Images (${processingProgress.validSilhouettes || 0} silhouettes)...`
                                        : processingProgress.stage === 'EXTRACTING_EMBEDDINGS'
                                        ? `Generating ByGaitLight (256D) embeddings (${processingProgress.validSequences || 0} sequences)...`
                                        : processingProgress.stage === 'MATCHING'
                                        ? 'Deduplicating embeddings & validating gallery...'
                                        : processingProgress.stage === 'PERSISTING'
                                        ? `Persisting Gallery (${processingProgress.embeddingsCommitted || 0} vectors committed)...`
                                        : processingProgress.stage === 'ENROLLING'
                                        ? 'Enrolling biometrics (ByGaitLight 256D + OSNet 512D)...'
                                        : 'Running Camera-Independent ByGaitLight (256D) Pipeline...'}
                                </p>

                                <div style={{ marginTop: '0.75rem', padding: '0.6rem 1rem', background: 'rgba(92,225,230,0.08)', borderRadius: '6px', border: '1px solid rgba(92,225,230,0.2)', textAlign: 'left', fontSize: '0.78rem', color: 'var(--text-primary)' }}>
                                    <div style={{ marginBottom: '0.2rem' }}>
                                        <strong style={{ color: '#5ce1e6' }}>Current Stage:</strong> {processingProgress.stage || 'PROCESSING'}
                                    </div>
                                    {videos.length > 0 && (
                                        <>
                                            <div style={{ marginBottom: '0.2rem' }}>
                                                <strong>Gait Sequences:</strong> {processingProgress.validSequences || gaitProgress?.valid_sequences || 0}
                                            </div>
                                            <div>
                                                <strong>256D Embeddings Committed:</strong> {processingProgress.embeddingsCommitted || gaitProgress?.embeddings_committed || 0}
                                            </div>
                                        </>
                                    )}
                                    {images.length > 0 && videos.length === 0 && (
                                        <div>
                                            <strong>Enrolling:</strong> {images.length} photo(s) into surveillance gallery
                                        </div>
                                    )}
                                </div>
                            </>
                        ) : (
                            <>
                                <h3 style={{ color: 'var(--ice)', marginTop: '1.25rem', fontSize: '1.25rem' }}>
                                    {uploadProgress.mediaType === 'video'
                                        ? 'Uploading Reference Video...'
                                        : uploadProgress.mediaType === 'image'
                                        ? 'Uploading Reference Photos...'
                                        : 'Uploading Case Data...'}
                                </h3>

                                <div className="progress-bar-container">
                                    <div
                                        className="progress-bar-fill"
                                        style={{ width: `${uploadProgress.percent}%` }}
                                    ></div>
                                </div>
                                <div className="progress-stats">
                                    <span>Upload: {uploadProgress.percent}%</span>
                                    <span>{uploadProgress.total > 0 ? `${formatBytes(uploadProgress.loaded)} / ${formatBytes(uploadProgress.total)}` : ''}</span>
                                </div>
                                {uploadProgress.speedMBs && uploadProgress.speedMBs !== '0.00' && (
                                    <div className="upload-meta-bar">
                                        <span>Speed: {uploadProgress.speedMBs} MB/s</span>
                                        <span>{uploadProgress.etaSeconds > 0 ? `ETA: ~${uploadProgress.etaSeconds}s` : ''}</span>
                                        <span className={`status-pill ${uploadProgress.connectionStatus || 'stable'}`}>
                                            {uploadProgress.connectionStatus === 'retrying' ? 'Retrying...' : 'Stable'}
                                        </span>
                                    </div>
                                )}

                                <p style={{ marginTop: '0.5rem', color: 'var(--text-secondary)', fontSize: '0.84rem' }}>
                                    {uploadProgress.label || 'Please wait, securing case files...'}
                                </p>
                                <button className="cancel-upload-btn" onClick={handleCancelUpload}>Cancel Upload</button>
                            </>
                        )}
                    </div>
                </div>
            )}

            {showSuccessModal && (
                <div className="modal-overlay" onClick={handleModalClose}>
                    <div className="success-modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: '540px', textAlign: 'center' }}>
                        <button className="modal-close-btn" onClick={handleModalClose}>
                            <XCircle size={28} fill="#E53935" color="#ffffff" />
                        </button>

                        {enrollmentWarning ? (
                            <>
                                <div style={{ display: 'inline-flex', padding: '12px', borderRadius: '50%', background: 'rgba(255, 171, 0, 0.15)', marginBottom: '1rem' }}>
                                    <AlertTriangle size={36} color="#ffab00" />
                                </div>
                                <h3 style={{ color: '#ffab00', marginBottom: '0.5rem' }}>
                                    Case Created — Biometric Enrollment Warning
                                </h3>
                                <p style={{ color: 'var(--ice)', fontWeight: 600, fontSize: '0.95rem', marginBottom: '0.75rem' }}>
                                    Case ID: {formData.caseId}
                                </p>
                                <div style={{
                                    padding: '0.85rem 1rem',
                                    background: 'rgba(255, 171, 0, 0.08)',
                                    border: '1px solid rgba(255, 171, 0, 0.3)',
                                    borderRadius: '8px',
                                    textAlign: 'left',
                                    fontSize: '0.82rem',
                                    color: 'var(--text-primary)',
                                    marginBottom: '1rem',
                                    lineHeight: 1.5
                                }}>
                                    <div style={{ fontWeight: 700, color: '#ffab00', marginBottom: '0.35rem' }}>
                                        Safety Policy Notice:
                                    </div>
                                    <div>{enrollmentWarning}</div>
                                    <div style={{ marginTop: '0.5rem', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                                        Case record and uploaded files have been safely stored, but biometric gait embeddings were not enrolled into the active surveillance gallery to prevent registering incorrect persons.
                                    </div>
                                </div>
                                <button
                                    onClick={handleModalClose}
                                    style={{
                                        padding: '0.6rem 1.8rem',
                                        background: 'var(--grad-primary)',
                                        border: 'none',
                                        borderRadius: '6px',
                                        color: '#fff',
                                        fontWeight: 700,
                                        cursor: 'pointer'
                                    }}
                                >
                                    Acknowledge & Close
                                </button>
                            </>
                        ) : enrollmentResult ? (
                            <>
                                <div style={{ display: 'inline-flex', padding: '12px', borderRadius: '50%', background: 'rgba(92, 225, 230, 0.15)', marginBottom: '1rem' }}>
                                    <CheckCircle size={36} color="#5ce1e6" />
                                </div>
                                <h3 style={{ color: 'var(--ice)', marginBottom: '0.5rem' }}>
                                    New Case Enrolled & Gallery Updated
                                </h3>
                                <p style={{ color: '#5ce1e6', fontWeight: 600, fontSize: '0.95rem', marginBottom: '0.75rem' }}>
                                    Case ID: {formData.caseId}
                                </p>
                                <div style={{
                                    padding: '0.85rem 1rem',
                                    background: 'rgba(92, 225, 230, 0.08)',
                                    border: '1px solid rgba(92, 225, 230, 0.25)',
                                    borderRadius: '8px',
                                    textAlign: 'left',
                                    fontSize: '0.82rem',
                                    color: 'var(--text-primary)',
                                    marginBottom: '1rem',
                                    lineHeight: 1.5
                                }}>
                                    <div style={{ marginBottom: '0.3rem' }}>
                                        <strong style={{ color: '#5ce1e6' }}>Gait Embeddings Committed:</strong>{' '}
                                        {enrollmentResult?.combinedResult?.gait_embeddings_added || 0} (256D ByGaitLight)
                                    </div>
                                    <div style={{ marginBottom: '0.3rem' }}>
                                        <strong>Appearance Embeddings:</strong>{' '}
                                        {enrollmentResult?.combinedResult?.appearance_embeddings_added || 0} (512D OSNet)
                                    </div>
                                    <div>
                                        <strong>Engine Status:</strong> COMPLETED &bull; Active Gallery Synchronized
                                    </div>
                                    {cloudSyncWarning && (
                                        <div style={{ marginTop: '0.35rem', color: '#ffab00', fontSize: '0.78rem' }}>
                                            <strong>Notice:</strong> {cloudSyncWarning}
                                        </div>
                                    )}
                                </div>
                                <button
                                    onClick={handleModalClose}
                                    style={{
                                        padding: '0.6rem 1.8rem',
                                        background: 'var(--grad-primary)',
                                        border: 'none',
                                        borderRadius: '6px',
                                        color: '#fff',
                                        fontWeight: 700,
                                        cursor: 'pointer'
                                    }}
                                >
                                    Done
                                </button>
                            </>
                        ) : (
                            <>
                                <h3 style={{ color: 'var(--ice)', marginBottom: '1rem' }}>
                                    New Case Added (Case ID: {formData.caseId})
                                </h3>
                                <button
                                    onClick={handleModalClose}
                                    style={{
                                        padding: '0.6rem 1.8rem',
                                        background: 'var(--grad-primary)',
                                        border: 'none',
                                        borderRadius: '6px',
                                        color: '#fff',
                                        fontWeight: 700,
                                        cursor: 'pointer'
                                    }}
                                >
                                    Done
                                </button>
                            </>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default ReportCase;
