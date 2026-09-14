import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { X, RefreshCw, Search, FolderOpen, ArrowLeft, ChevronRight } from 'lucide-react';
import { API_BASE, getAuthHeaders } from '../config/apiConfig';
import './CaseDossierModal.css';

/**
 * Format bytes to human-readable string (KB, MB, GB).
 */
function formatBytes(bytes) {
    if (!bytes || bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

/**
 * Get emoji icon for a file based on its MIME type.
 */
function getFileIcon(mime) {
    if (!mime) return '📄';
    if (mime.startsWith('video/')) return '🎬';
    if (mime.startsWith('image/')) return '🖼️';
    if (mime.includes('json')) return '📋';
    return '📄';
}

/**
 * Get CSS class for case type badge.
 */
function getTypeBadgeClass(caseType) {
    const t = (caseType || '').toLowerCase();
    if (t === 'kidnapping') return 'kidnapping';
    if (t === 'missing') return 'missing';
    return 'missing';
}

/**
 * Get CSS class for status badge.
 */
function getStatusBadgeClass(status) {
    const s = (status || '').toLowerCase();
    if (s === 'found' || s === 'closed') return 'found';
    if (s === 'cold') return 'cold';
    return 'investigating';
}

/**
 * Get current timestamp formatted for display.
 */
function getCurrentTimestamp() {
    return new Date().toLocaleString('en-US', {
        year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit', second: '2-digit',
        hour12: false
    });
}

const FILTER_OPTIONS = ['ALL', 'MISSING', 'KIDNAPPING', 'INVESTIGATING', 'FOUND', 'COLD'];

/* ────────────────────────────────────────────────────────────────────────── */
/*  CaseDossierModal                                                         */
/* ────────────────────────────────────────────────────────────────────────── */

const CaseDossierModal = ({ isOpen, onClose }) => {
    const [dossiers, setDossiers] = useState([]);
    const [loading, setLoading] = useState(false);
    const [syncing, setSyncing] = useState(false);
    const [searchQuery, setSearchQuery] = useState('');
    const [activeFilter, setActiveFilter] = useState('ALL');
    const [selectedDossier, setSelectedDossier] = useState(null);
    const [activeTab, setActiveTab] = useState('files');
    const [currentTime, setCurrentTime] = useState(getCurrentTimestamp());

    // Fetch dossiers from backend
    const fetchDossiers = useCallback(async (showSync = false) => {
        if (showSync) setSyncing(true);
        else setLoading(true);

        try {
            const res = await fetch(`${API_BASE}/api/v1/cases/dossiers`, {
                headers: getAuthHeaders(),
            });
            if (res.ok) {
                const data = await res.json();
                setDossiers(data.dossiers || []);
            } else {
                console.error('[ARGUS] Failed to fetch dossiers:', res.status);
            }
        } catch (err) {
            console.error('[ARGUS] Dossier fetch error:', err);
        } finally {
            setLoading(false);
            setSyncing(false);
        }
    }, []);

    // Load dossiers when modal opens
    useEffect(() => {
        if (isOpen) {
            fetchDossiers();
        } else {
            // Reset state when modal closes
            setSelectedDossier(null);
            setActiveTab('files');
            setSearchQuery('');
            setActiveFilter('ALL');
        }
    }, [isOpen, fetchDossiers]);

    // Update clock
    useEffect(() => {
        if (!isOpen) return;
        const interval = setInterval(() => setCurrentTime(getCurrentTimestamp()), 1000);
        return () => clearInterval(interval);
    }, [isOpen]);

    // Close on Escape
    useEffect(() => {
        if (!isOpen) return;
        const handler = (e) => {
            if (e.key === 'Escape') {
                if (selectedDossier) setSelectedDossier(null);
                else onClose();
            }
        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, [isOpen, selectedDossier, onClose]);

    // Filter and search dossiers
    const filteredDossiers = useMemo(() => {
        let result = [...dossiers];

        if (activeFilter !== 'ALL') {
            result = result.filter(d => {
                const filterLower = activeFilter.toLowerCase();
                const caseType = (d.case_type || '').toLowerCase();
                const status = (d.status || '').toLowerCase();
                return caseType === filterLower || status === filterLower;
            });
        }

        if (searchQuery.trim()) {
            const q = searchQuery.toLowerCase();
            result = result.filter(d =>
                (d.case_id || '').toLowerCase().includes(q) ||
                (d.person_name || '').toLowerCase().includes(q) ||
                (d.nic || '').toLowerCase().includes(q) ||
                (d.location?.name || '').toLowerCase().includes(q)
            );
        }

        return result;
    }, [dossiers, activeFilter, searchQuery]);

    if (!isOpen) return null;

    const handleSync = () => fetchDossiers(true);
    const handleOverlayClick = (e) => {
        if (e.target === e.currentTarget) onClose();
    };

    return (
        <div className="dossier-modal-overlay" onClick={handleOverlayClick}>
            <div className="dossier-modal-container" onClick={(e) => e.stopPropagation()}>
                {/* ── Header ── */}
                <div className="dossier-header">
                    <div className="dossier-header-top">
                        <div className="dossier-header-title">
                            <span className="dossier-icon">📂</span>
                            <div>
                                <h2>CLASSIFIED CASE DOSSIERS // FOLDER ARCHIVE</h2>
                                <span className="dossier-timestamp">SYSTEM TIME: {currentTime}</span>
                            </div>
                        </div>
                        <button className="dossier-close-btn" onClick={onClose}>
                            <X size={14} /> CLOSE
                        </button>
                    </div>

                    <div className="dossier-controls">
                        <input
                            className="dossier-search"
                            type="text"
                            placeholder="Search by Case ID, Person Name, NIC, Location..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                        />
                        <div className="dossier-filter-pills">
                            {FILTER_OPTIONS.map(f => (
                                <button
                                    key={f}
                                    className={`dossier-filter-pill ${activeFilter === f ? 'active' : ''}`}
                                    onClick={() => setActiveFilter(f)}
                                >
                                    {f}
                                </button>
                            ))}
                        </div>
                        <button
                            className={`dossier-sync-btn ${syncing ? 'syncing' : ''}`}
                            onClick={handleSync}
                            disabled={syncing}
                        >
                            <RefreshCw size={13} className={syncing ? 'spin-icon' : ''} />
                            {syncing ? 'SYNCING...' : 'SYNC'}
                        </button>
                    </div>
                </div>

                {/* ── Body ── */}
                <div className="dossier-body">
                    {loading ? (
                        <div className="dossier-loading">
                            <div className="spinner"></div>
                            <p>Loading case dossiers...</p>
                        </div>
                    ) : selectedDossier ? (
                        <DossierDetailView
                            dossier={selectedDossier}
                            onBack={() => setSelectedDossier(null)}
                            activeTab={activeTab}
                            setActiveTab={setActiveTab}
                        />
                    ) : filteredDossiers.length === 0 ? (
                        <div className="dossier-empty">
                            <span className="dossier-empty-icon">📁</span>
                            <p>No case dossiers found{searchQuery ? ` matching "${searchQuery}"` : ''}</p>
                        </div>
                    ) : (
                        <div className="dossier-folder-grid">
                            {filteredDossiers.map(d => (
                                <FolderCard
                                    key={d.case_id}
                                    dossier={d}
                                    onClick={() => {
                                        setSelectedDossier(d);
                                        setActiveTab('files');
                                    }}
                                />
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};


/* ────────────────────────────────────────────────────────────────────────── */
/*  FolderCard — One case folder in the grid                                 */
/* ────────────────────────────────────────────────────────────────────────── */

const FolderCard = ({ dossier, onClick }) => {
    const d = dossier;
    const filesCount = d.files_inventory?.total_files || 0;
    const totalSize = d.files_inventory?.total_size_bytes || 0;
    const mediaCount = (d.media?.photos_count || 0) + (d.media?.videos_count || 0);
    const isEnrolled = d.biometrics?.enrolled || false;
    const gaitCount = d.biometrics?.gait_embeddings_count || 0;

    return (
        <div className="dossier-folder-card" onClick={onClick} role="button" tabIndex={0}>
            <div className="dossier-folder-card-header">
                <div className="folder-icon-name">
                    <span className="folder-emoji">📂</span>
                    <div className="folder-name-group">
                        <span className="case-id-label">{d.case_id}</span>
                        <span className="person-name">{d.person_name || 'Unknown'}</span>
                    </div>
                </div>
                <div className="dossier-folder-card-badges">
                    <span className={`dossier-badge ${getTypeBadgeClass(d.case_type)}`}>
                        {d.case_type || 'Unknown'}
                    </span>
                    <span className={`dossier-badge ${getStatusBadgeClass(d.status)}`}>
                        {d.status || 'Unknown'}
                    </span>
                </div>
            </div>

            <div className="dossier-folder-card-meta">
                <span className="dossier-meta-item">
                    <span className="meta-icon">📄</span>
                    {filesCount} files
                </span>
                <span className="dossier-meta-item">
                    <span className="meta-icon">💾</span>
                    {formatBytes(totalSize)}
                </span>
                {mediaCount > 0 && (
                    <span className="dossier-meta-item">
                        <span className="meta-icon">🎬</span>
                        {mediaCount} media
                    </span>
                )}
                <span className="dossier-meta-item">
                    <span className={`dossier-enrolled-dot ${isEnrolled ? 'active' : 'pending'}`}></span>
                    {isEnrolled ? `256D×${gaitCount}` : 'Not Enrolled'}
                </span>
            </div>
        </div>
    );
};


/* ────────────────────────────────────────────────────────────────────────── */
/*  DossierDetailView — Inside a case folder                                 */
/* ────────────────────────────────────────────────────────────────────────── */

const TABS = [
    { id: 'files', label: 'Folder Files', icon: '📁' },
    { id: 'intel', label: 'Case Intelligence', icon: '👤' },
    { id: 'media', label: 'Media & Evidence', icon: '🎬' },
    { id: 'bio', label: '256D Biometrics', icon: '🧬' },
];

const DossierDetailView = ({ dossier, onBack, activeTab, setActiveTab }) => {
    const d = dossier;

    return (
        <div className="dossier-detail-view">
            {/* Breadcrumb */}
            <div className="dossier-breadcrumb">
                <button className="dossier-breadcrumb-back" onClick={onBack}>
                    <ArrowLeft size={14} /> BACK
                </button>
                <span className="dossier-breadcrumb-text">
                    📁 All Cases <ChevronRight size={12} style={{ verticalAlign: 'middle' }} />{' '}
                    <span className="crumb-active">📂 {d.folder_name || d.case_id}</span>
                </span>
            </div>

            {/* Tabs */}
            <div className="dossier-tabs">
                {TABS.map(tab => (
                    <button
                        key={tab.id}
                        className={`dossier-tab ${activeTab === tab.id ? 'active' : ''}`}
                        onClick={() => setActiveTab(tab.id)}
                    >
                        <span>{tab.icon}</span> {tab.label}
                    </button>
                ))}
            </div>

            {/* Tab Content */}
            <div className="dossier-tab-content">
                {activeTab === 'files' && <FilesTab dossier={d} />}
                {activeTab === 'intel' && <IntelTab dossier={d} />}
                {activeTab === 'media' && <MediaTab dossier={d} />}
                {activeTab === 'bio' && <BiometricsTab dossier={d} />}
            </div>
        </div>
    );
};


/* ── Files Tab ── */
const FilesTab = ({ dossier }) => {
    const files = dossier.files_inventory?.files || [];

    if (files.length === 0) {
        return (
            <div className="dossier-no-data">
                <span className="no-data-icon">📁</span>
                <p>No files in this dossier folder</p>
            </div>
        );
    }

    return (
        <div className="dossier-file-list">
            {files.map((f, idx) => (
                <div key={idx} className="dossier-file-row">
                    <span className="dossier-file-icon">{getFileIcon(f.mime_type)}</span>
                    <div className="dossier-file-info">
                        <div className="dossier-file-name" title={f.rel_path}>{f.name}</div>
                        <div className="dossier-file-meta">
                            {f.rel_path} · {formatBytes(f.size_bytes)} · {f.mime_type}
                        </div>
                    </div>
                    <div className="dossier-file-actions">
                        <a
                            className="dossier-file-action-btn"
                            href={`${API_BASE}${f.download_url}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            onClick={(e) => e.stopPropagation()}
                        >
                            VIEW
                        </a>
                    </div>
                </div>
            ))}
        </div>
    );
};


/* ── Case Intelligence Tab ── */
const IntelTab = ({ dossier }) => {
    const d = dossier;
    const loc = d.location || {};

    return (
        <div className="dossier-intel-grid">
            <div className="dossier-intel-card">
                <h4>👤 Personal Details</h4>
                <div className="dossier-intel-row">
                    <span className="dossier-intel-label">Name</span>
                    <span className="dossier-intel-value">{d.person_name || '—'}</span>
                </div>
                <div className="dossier-intel-row">
                    <span className="dossier-intel-label">Case ID</span>
                    <span className="dossier-intel-value">{d.case_id || '—'}</span>
                </div>
                <div className="dossier-intel-row">
                    <span className="dossier-intel-label">NIC</span>
                    <span className="dossier-intel-value">{d.nic || '—'}</span>
                </div>
                <div className="dossier-intel-row">
                    <span className="dossier-intel-label">Age</span>
                    <span className="dossier-intel-value">{d.age || '—'}</span>
                </div>
                <div className="dossier-intel-row">
                    <span className="dossier-intel-label">Gender</span>
                    <span className="dossier-intel-value">{d.gender || '—'}</span>
                </div>
            </div>

            <div className="dossier-intel-card">
                <h4>📋 Case Details</h4>
                <div className="dossier-intel-row">
                    <span className="dossier-intel-label">Type</span>
                    <span className="dossier-intel-value">{d.case_type || '—'}</span>
                </div>
                <div className="dossier-intel-row">
                    <span className="dossier-intel-label">Status</span>
                    <span className="dossier-intel-value">{d.status || '—'}</span>
                </div>
                <div className="dossier-intel-row">
                    <span className="dossier-intel-label">Priority</span>
                    <span className="dossier-intel-value">{d.priority || '—'}</span>
                </div>
                <div className="dossier-intel-row">
                    <span className="dossier-intel-label">Reported</span>
                    <span className="dossier-intel-value">{d.reported_at || '—'}</span>
                </div>
                <div className="dossier-intel-row">
                    <span className="dossier-intel-label">Folder</span>
                    <span className="dossier-intel-value">{d.folder_name || '—'}</span>
                </div>
            </div>

            <div className="dossier-intel-card">
                <h4>📍 Last Seen Location</h4>
                <div className="dossier-intel-row">
                    <span className="dossier-intel-label">Place</span>
                    <span className="dossier-intel-value">{loc.name || '—'}</span>
                </div>
                <div className="dossier-intel-row">
                    <span className="dossier-intel-label">Latitude</span>
                    <span className="dossier-intel-value">{loc.lat != null ? loc.lat : '—'}</span>
                </div>
                <div className="dossier-intel-row">
                    <span className="dossier-intel-label">Longitude</span>
                    <span className="dossier-intel-value">{loc.lng != null ? loc.lng : '—'}</span>
                </div>
                <div className="dossier-intel-row">
                    <span className="dossier-intel-label">Source</span>
                    <span className="dossier-intel-value">{loc.source || '—'}</span>
                </div>
            </div>
        </div>
    );
};


/* ── Media Tab ── */
const MediaTab = ({ dossier }) => {
    const videos = dossier.media?.videos || [];
    const photos = dossier.media?.photos || [];
    const hasMedia = videos.length > 0 || photos.length > 0;

    if (!hasMedia) {
        return (
            <div className="dossier-no-data">
                <span className="no-data-icon">🎬</span>
                <p>No media files enrolled for this case</p>
            </div>
        );
    }

    return (
        <div>
            {videos.length > 0 && (
                <div className="dossier-media-section">
                    <h4>🎬 Reference Videos ({videos.length})</h4>
                    {videos.map((v, idx) => (
                        <div key={idx} style={{ marginBottom: '1rem' }}>
                            <video
                                className="dossier-video-player"
                                controls
                                preload="metadata"
                                src={`${API_BASE}${v.download_url}`}
                            >
                                Your browser does not support video playback.
                            </video>
                            <div style={{ fontSize: '0.72rem', color: '#78909C', marginTop: '0.3rem', fontWeight: 600 }}>
                                {v.name} · {formatBytes(v.size_bytes)}
                            </div>
                        </div>
                    ))}
                </div>
            )}

            {photos.length > 0 && (
                <div className="dossier-media-section">
                    <h4>🖼️ Reference Photos ({photos.length})</h4>
                    <div className="dossier-photo-grid">
                        {photos.map((p, idx) => (
                            <div key={idx} className="dossier-photo-thumb">
                                <img
                                    src={`${API_BASE}${p.download_url}`}
                                    alt={p.name}
                                    loading="lazy"
                                />
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};


/* ── Biometrics Tab ── */
const BiometricsTab = ({ dossier }) => {
    const bio = dossier.biometrics || {};
    const isEnrolled = bio.enrolled || false;
    const isActive = bio.active_surveillance || false;

    return (
        <div>
            <div className="dossier-bio-grid">
                <div className="dossier-bio-stat">
                    <span className="bio-stat-icon">🧬</span>
                    <span className="bio-stat-value">{bio.gait_embeddings_count || 0}</span>
                    <span className="bio-stat-label">ByGaitLight 256D Embeddings</span>
                </div>
                <div className="dossier-bio-stat">
                    <span className="bio-stat-icon">👁️</span>
                    <span className="bio-stat-value">{bio.appearance_embeddings_count || 0}</span>
                    <span className="bio-stat-label">OSNet 512D Appearance</span>
                </div>
                <div className="dossier-bio-stat">
                    <span className="bio-stat-icon">🔐</span>
                    <span className="bio-stat-value">{bio.gallery_status || 'N/A'}</span>
                    <span className="bio-stat-label">Gallery Status</span>
                </div>
                <div className="dossier-bio-stat">
                    <span className="bio-stat-icon">🏷️</span>
                    <span className="bio-stat-value" style={{ fontSize: '0.82rem' }}>{bio.model || 'N/A'}</span>
                    <span className="bio-stat-label">Model Architecture</span>
                </div>
            </div>

            <div className="dossier-bio-status-bar">
                <div className={`status-indicator ${isActive ? 'active' : 'inactive'}`}>
                    <span className={`pulse-dot ${isActive ? 'active' : 'inactive'}`}></span>
                    {isActive ? 'ACTIVE SURVEILLANCE' : 'NOT IN SURVEILLANCE'}
                </div>
                <span className="status-label">
                    {isEnrolled
                        ? `Subject enrolled with ${bio.gait_embeddings_count || 0} gait embeddings — Live matching enabled`
                        : 'Subject not yet enrolled in biometric gallery'
                    }
                </span>
            </div>

            {bio.job_id && (
                <div className="dossier-bio-status-bar" style={{ marginTop: '0.5rem' }}>
                    <span className="status-label" style={{ color: '#90A4AE' }}>
                        📎 Reference Job: <span style={{ color: '#e8ecf1', fontFamily: "'JetBrains Mono', monospace", fontSize: '0.75rem' }}>{bio.job_id}</span>
                    </span>
                </div>
            )}
        </div>
    );
};

export default CaseDossierModal;
