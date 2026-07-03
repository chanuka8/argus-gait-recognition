import React, { useState, useEffect } from 'react';
import { Search, Trash2, Briefcase, X, Eye, EyeOff } from 'lucide-react';
import AdminHeader from './AdminHeader';
import { db, storage } from '../firebaseConfig';
import { collection, getDocs, doc, getDoc, deleteDoc, query, where } from 'firebase/firestore';
import { ref, deleteObject } from 'firebase/storage';
import { useAuth } from '../contexts/AuthContext';
import { addLog } from '../utils/logService';
import './CasesManagement.css';

const CasesManagement = () => {
    const { currentUser } = useAuth();
    const [cases, setCases] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');

    // Deletion states
    const [deletingCase, setDeletingCase] = useState(null);
    const [confirmPassword, setConfirmPassword] = useState('');
    const [showConfirmPassword, setShowConfirmPassword] = useState(false);
    const [passwordError, setPasswordError] = useState('');
    const [isAuthorizing, setIsAuthorizing] = useState(false);
    const [isDeleteSuccess, setIsDeleteSuccess] = useState(false);

    const fetchCases = async () => {
        try {
            setIsLoading(true);
            const snapshot = await getDocs(collection(db, 'victims'));
            const casesList = [];
            snapshot.forEach((doc) => {
                const data = doc.data();
                casesList.push({
                    id: doc.id,
                    caseId: data.caseId || doc.id,
                    caseType: data.caseType || 'N/A',
                    status: data.status || 'Investigating'
                });
            });
            setCases(casesList);
        } catch (error) {
            console.error('Error fetching cases:', error);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchCases();
    }, []);

    const handleDeleteClick = (c) => {
        setDeletingCase(c);
        setConfirmPassword('');
        setShowConfirmPassword(false);
        setPasswordError('');
        setIsDeleteSuccess(false);
    };

    const handleCloseModal = () => {
        if (isAuthorizing || isDeleteSuccess) return;
        setDeletingCase(null);
        setConfirmPassword('');
        setShowConfirmPassword(false);
        setPasswordError('');
    };

    const handleConfirmDelete = async (e) => {
        e.preventDefault();
        setPasswordError('');

        if (!confirmPassword.trim()) {
            setPasswordError('Administrator password is required.');
            return;
        }

        setIsAuthorizing(true);

        try {
            // Verify admin password by querying username
            const adminQuery = query(
                collection(db, 'admins'),
                where('username', '==', currentUser?.username || '')
            );
            const querySnapshot = await getDocs(adminQuery);

            if (querySnapshot.empty) {
                setPasswordError('Administrator credentials record not found in system database.');
                setIsAuthorizing(false);
                return;
            }

            const adminDoc = querySnapshot.docs[0];
            const correctPassword = adminDoc.data().password;
            if (confirmPassword !== correctPassword) {
                setPasswordError('Incorrect password. Authorization failed.');
                setIsAuthorizing(false);
                return;
            }

            // Password is correct, perform cascade deletion:
            // Fetch associated media documents to get storage files
            const mediaRef = doc(db, 'person_media', deletingCase.id);
            const mediaSnap = await getDoc(mediaRef);

            if (mediaSnap.exists()) {
                const mediaData = mediaSnap.data();
                const imageUrls = mediaData.imageUrls || [];
                const videoUrls = mediaData.videoUrls || [];

                // Delete image files from Firebase Storage
                for (const url of imageUrls) {
                    if (url.includes('firebasestorage.googleapis.com')) {
                        try {
                            const fileRef = ref(storage, url);
                            await deleteObject(fileRef);
                            console.log(`Deleted case storage file: ${url}`);
                        } catch (storageErr) {
                            console.error(`Failed to delete storage file: ${url}`, storageErr);
                        }
                    }
                }

                // Delete video files from Firebase Storage
                for (const url of videoUrls) {
                    if (url.includes('firebasestorage.googleapis.com')) {
                        try {
                            const fileRef = ref(storage, url);
                            await deleteObject(fileRef);
                            console.log(`Deleted case storage video: ${url}`);
                        } catch (storageErr) {
                            console.error(`Failed to delete storage video: ${url}`, storageErr);
                        }
                    }
                }

                // Delete person_media Firestore document
                await deleteDoc(mediaRef);
            }

            // Delete main victims case document
            const victimRef = doc(db, 'victims', deletingCase.id);
            await deleteDoc(victimRef);

            // Log the critical action in system logs
            addLog(
                'critical',
                `Case ${deletingCase.caseId} permanently removed`,
                `Admin @${currentUser.username} deleted case ${deletingCase.caseId} (${deletingCase.caseType}) and all associated media archives from the system.`,
                currentUser.username
            );

            // Update local state list
            setCases(prev => prev.filter(c => c.id !== deletingCase.id));

            // Trigger success state
            setIsDeleteSuccess(true);

            // Reset and close after delay
            setTimeout(() => {
                setDeletingCase(null);
                setConfirmPassword('');
                setIsDeleteSuccess(false);
            }, 1500);

        } catch (error) {
            console.error('Error deleting case:', error);
            setPasswordError(`Deletion failed: ${error.message || error.toString()}`);
        } finally {
            setIsAuthorizing(false);
        }
    };



    // Filter cases by ID or type
    const filteredCases = cases.filter(c => {
        const term = searchTerm.toLowerCase();
        return String(c.caseId).toLowerCase().includes(term) ||
            String(c.caseType).toLowerCase().includes(term) ||
            String(c.status).toLowerCase().includes(term);
    });

    const getStatusColor = (status) => {
        switch (status?.toLowerCase()) {
            case 'cold': return 'status-cold';
            case 'investigating': return 'status-investigating';
            case 'found': return 'status-found';
            case 'closed': return 'status-closed';
            default: return 'status-default';
        }
    };

    return (
        <div className="cases-mgmt-page">
            <AdminHeader />

            <main className="cases-mgmt-content">
                <div className="cases-mgmt-header-row">
                    <div className="title-group">
                        <h1>Case Records</h1>
                    </div>
                </div>

                <div className="search-controls">
                    <div className="cases-search-bar">
                        <Search size={18} color="var(--text-muted)" />
                        <input
                            type="text"
                            placeholder="Search cases by Case ID, Type, or Status..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                        />
                    </div>
                </div>

                <div className="cases-table-container">
                    <table className="cases-table">
                        <thead>
                            <tr>
                                <th>Case ID</th>
                                <th>Case Type</th>
                                <th>Status</th>
                                <th style={{ textAlign: 'center' }}>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {isLoading ? (
                                <tr>
                                    <td colSpan="4" className="no-cases-row">
                                        <div className="loading-spinner-small"></div>
                                        <p style={{ marginTop: '0.5rem' }}>Loading active case registries...</p>
                                    </td>
                                </tr>
                            ) : filteredCases.length === 0 ? (
                                <tr>
                                    <td colSpan="4" className="no-cases-row">
                                        <Search size={30} color="var(--text-muted)" />
                                        <p>No matching case logs found.</p>
                                    </td>
                                </tr>
                            ) : (
                                filteredCases.map((c) => (
                                    <tr key={c.id}>
                                        <td className="case-id-cell">{c.caseId}</td>
                                        <td className="case-type-cell">
                                            <span className="type-label">{c.caseType}</span>
                                        </td>
                                        <td>
                                            <span className={`case-status-badge ${getStatusColor(c.status)}`}>
                                                <span className="status-dot"></span>
                                                {c.status}
                                            </span>
                                        </td>
                                        <td>
                                            <div className="actions-cell">
                                                <button
                                                    className="delete-case-btn"
                                                    onClick={() => handleDeleteClick(c)}
                                                    title="Delete Case Log"
                                                >
                                                    <Trash2 size={16} />
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            </main>

            {/* Password Verification Modal */}
            {deletingCase && (
                <div
                    className="delete-modal-overlay"
                    onClick={(e) => {
                        if (e.target === e.currentTarget) {
                            handleCloseModal();
                        }
                    }}
                >
                    <div className="delete-modal-card">
                        {isDeleteSuccess ? (
                            <div className="deletion-success-animation">
                                <div className="trash-circle">
                                    <Trash2 size={36} className="trash-svg" />
                                </div>
                                <h3>Case Log Removed!</h3>
                                <p>Case and associated media files were removed permanently.</p>
                            </div>
                        ) : (
                            <>
                                <div className="modal-header">
                                    <div className="header-title-flex">
                                        <Briefcase size={20} className="modal-header-icon" />
                                        <h2>Authorize Case Purge</h2>
                                    </div>
                                    <button className="modal-close-btn" onClick={handleCloseModal} disabled={isAuthorizing}>
                                        <X size={20} />
                                    </button>
                                </div>

                                {passwordError && <div className="form-error-banner">{passwordError}</div>}

                                <form onSubmit={handleConfirmDelete} className="delete-case-form">
                                    <div className="warning-banner">
                                        <p>
                                            <strong>CRITICAL:</strong> You are authorizing the permanent deletion of case <strong>{deletingCase.caseId}</strong>.
                                            This action is irreversible. All databases entries, metadata connections, and stored files in Firebase Storage will be deleted.
                                        </p>
                                    </div>

                                    <div className="form-field">
                                        <label>Confirm Administrator Password</label>
                                        <div className="password-input-wrapper">
                                            <input
                                                type={showConfirmPassword ? "text" : "password"}
                                                value={confirmPassword}
                                                onChange={(e) => setConfirmPassword(e.target.value)}
                                                placeholder="••••••••"
                                                disabled={isAuthorizing}
                                                required
                                            />
                                            <button
                                                type="button"
                                                className="password-toggle-btn"
                                                onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                                            >
                                                {showConfirmPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                                            </button>
                                        </div>
                                    </div>

                                    <div className="modal-actions">
                                        <button
                                            type="button"
                                            className="modal-cancel-btn"
                                            onClick={handleCloseModal}
                                            disabled={isAuthorizing}
                                        >
                                            Cancel
                                        </button>
                                        <button
                                            type="submit"
                                            className="modal-confirm-btn"
                                            disabled={isAuthorizing}
                                        >
                                            {isAuthorizing ? 'Purging Case...' : 'Authorize Deletion'}
                                        </button>
                                    </div>
                                </form>
                            </>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default CasesManagement;
