import React, { useState, useEffect, useRef } from 'react';
import { UserPlus, Search, ToggleLeft, ToggleRight, Trash2, X, Camera, Eye, EyeOff } from 'lucide-react';
import AdminHeader from './AdminHeader';
import { db, storage } from '../firebaseConfig';
import { collection, getDocs, doc, getDoc, setDoc, deleteDoc, updateDoc } from 'firebase/firestore';
import { ref, uploadBytesResumable, getDownloadURL, deleteObject } from 'firebase/storage';
import { addLog } from '../utils/logService';
import './UserManagement.css';

const validatePassword = (password) => {
    const minLength = 8;
    const hasUpperCase = /[A-Z]/.test(password);
    const hasLowerCase = /[a-z]/.test(password);
    const hasNumber = /[0-9]/.test(password);
    const hasSpecialChar = /[!@#$%^&*(),.?":{}|<>]/.test(password);

    if (password.length < minLength) {
        return 'Password must be at least 8 characters long.';
    }
    if (!hasUpperCase) {
        return 'Password must contain at least one uppercase letter.';
    }
    if (!hasLowerCase) {
        return 'Password must contain at least one lowercase letter.';
    }
    if (!hasNumber) {
        return 'Password must contain at least one number.';
    }
    if (!hasSpecialChar) {
        return 'Password must contain at least one special character.';
    }
    return null;
};

const UserManagement = () => {
    const [users, setUsers] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');
    const [showAddModal, setShowAddModal] = useState(false);
    const fileInputRef = useRef(null);

    // Add user form states
    const [newName, setNewName] = useState('');
    const [newUsername, setNewUsername] = useState('');
    const [newRole, setNewRole] = useState('Investigator');
    const [newNic, setNewNic] = useState('');
    const [imageFile, setImageFile] = useState(null);
    const [newPassword, setNewPassword] = useState('');
    const [formError, setFormError] = useState('');
    const [isRegisterSuccess, setIsRegisterSuccess] = useState(false);
    const [isUploading, setIsUploading] = useState(false);
    const [showNewPassword, setShowNewPassword] = useState(false);
    const [editingUser, setEditingUser] = useState(null);

    const handleImageChange = (e) => {
        if (e.target.files && e.target.files[0]) {
            setImageFile(e.target.files[0]);
        } else {
            setImageFile(null);
        }
    };

    const handleEditClick = (user) => {
        setEditingUser(user);
        setNewName(user.name);
        setNewUsername(user.username);
        setNewRole(user.role === 'Admin' ? 'Admin' : 'Investigator');
        setNewNic(user.nic);
        setNewPassword('');
        setImageFile(null);
        setShowAddModal(true);
    };

    const handleCloseModal = () => {
        if (isUploading || isRegisterSuccess) return;
        setNewName('');
        setNewUsername('');
        setNewRole('Investigator');
        setNewNic('');
        setImageFile(null);
        setNewPassword('');
        setFormError('');
        setEditingUser(null);
        setIsRegisterSuccess(false);
        setShowNewPassword(false);
        setShowAddModal(false);
    };

    // Fetch operators from both firestore collections on mount
    const fetchOperators = async () => {
        try {
            setIsLoading(true);
            const mergedUsers = [];

            // 1. Fetch from 'admins'
            const adminSnapshot = await getDocs(collection(db, 'admins'));
            adminSnapshot.forEach((doc) => {
                const data = doc.data();
                mergedUsers.push({
                    id: doc.id,
                    name: data.name,
                    username: data.username,
                    role: 'Admin',
                    nic: data.nic || '',
                    image: data.image || '',
                    status: data.status || 'Active',
                    lastLogin: data.lastLogin || 'Never'
                });
            });

            // 2. Fetch from 'investigators'
            const invSnapshot = await getDocs(collection(db, 'investigators'));
            invSnapshot.forEach((doc) => {
                const data = doc.data();
                mergedUsers.push({
                    id: doc.id,
                    name: data.name,
                    username: data.username,
                    role: 'Investigator',
                    nic: data.nic || '',
                    image: data.image || '',
                    status: data.status || 'Active',
                    lastLogin: data.lastLogin || 'Never'
                });
            });

            setUsers(mergedUsers);
        } catch (error) {
            console.error('Error fetching operators:', error);
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        fetchOperators();
    }, []);

    // Handle adding users
    const handleAddUser = async (e) => {
        e.preventDefault();
        setFormError('');

        if (!newName.trim() || !newUsername.trim() || !newPassword.trim() || !newNic.trim()) {
            setFormError('All fields are required.');
            return;
        }

        // Validate password rules
        const passwordError = validatePassword(newPassword);
        if (passwordError) {
            setFormError(passwordError);
            return;
        }

        // Validate that username contains role identifiers
        let finalUsername = newUsername.trim().toLowerCase();
        if (newRole === 'Admin' && !finalUsername.includes('admin')) {
            finalUsername = `admin_${finalUsername}`;
        } else if (newRole === 'Investigator' && !finalUsername.includes('inv')) {
            finalUsername = `inv_${finalUsername}`;
        }

        // Check duplicates locally first
        if (users.some(u => u.username.toLowerCase() === finalUsername.toLowerCase())) {
            setFormError('Operator with this username already exists.');
            return;
        }

        setIsUploading(true);

        try {
            // Upload to storage if file is present
            let finalImageUrl = '';
            if (imageFile) {
                const fileRef = ref(storage, `profiles/${finalUsername}/${imageFile.name}`);
                const uploadTask = uploadBytesResumable(fileRef, imageFile);
                await new Promise((resolve, reject) => {
                    uploadTask.on('state_changed', null, reject, () => resolve());
                });
                finalImageUrl = await getDownloadURL(fileRef);
            } else {
                // Default fallback avatar seed using the username
                finalImageUrl = `https://api.dicebear.com/7.x/bottts/svg?seed=${finalUsername}`;
            }

            const roleLower = newRole.toLowerCase(); // 'admin' or 'investigator'
            const targetCollection = roleLower === 'admin' ? 'admins' : 'investigators';
            const docId = finalUsername;

            const docRef = doc(db, targetCollection, docId);
            const newDocData = {
                name: newName,
                username: finalUsername,
                password: newPassword,
                nic: newNic,
                image: finalImageUrl,
                role: roleLower,
                status: 'Active',
                lastLogin: 'Never'
            };

            await setDoc(docRef, newDocData);

            // Add to local state to avoid refetching
            setUsers(prev => [
                ...prev,
                {
                    id: docId,
                    name: newName,
                    username: finalUsername,
                    role: newRole,
                    nic: newNic,
                    image: finalImageUrl,
                    status: 'Active',
                    lastLogin: 'Never'
                }
            ]);

            // Trigger success animation
            setIsRegisterSuccess(true);

            // Wait 1.8s to show success state before resetting/closing
            setTimeout(() => {
                setNewName('');
                setNewUsername('');
                setNewRole('Investigator');
                setNewNic('');
                setImageFile(null);
                setNewPassword('');
                setIsRegisterSuccess(false);
                setShowNewPassword(false);
                setShowAddModal(false);
            }, 1800);

            // Record user creation in system logs
            addLog('info', `New operator registered: ${finalUsername}`, `Operator ${newName} was created with role ${newRole}. NIC: ${newNic}. Account status: Active.`, 'admin');
        } catch (error) {
            console.error('Error creating operator:', error);
            setFormError(`Failed to save operator to database: ${error.message || error.toString()}`);
        } finally {
            setIsUploading(false);
        }
    };

    const handleUpdateUser = async () => {
        setFormError('');

        if (!newName.trim() || !newNic.trim()) {
            setFormError('Name and NIC fields are required.');
            return;
        }

        // Validate password rules only if a new one is typed
        if (newPassword.trim()) {
            const passwordError = validatePassword(newPassword);
            if (passwordError) {
                setFormError(passwordError);
                return;
            }
        }

        setIsUploading(true);

        try {
            // Find existing image URL
            let finalImageUrl = editingUser.image || '';

            // Upload to storage if new file is selected
            if (imageFile) {
                const fileRef = ref(storage, `profiles/${editingUser.username}/${imageFile.name}`);
                const uploadTask = uploadBytesResumable(fileRef, imageFile);
                await new Promise((resolve, reject) => {
                    uploadTask.on('state_changed', null, reject, () => resolve());
                });
                finalImageUrl = await getDownloadURL(fileRef);

                // Try to delete old image if it was a Storage file and different
                if (editingUser.image && editingUser.image.includes('firebasestorage.googleapis.com') && editingUser.image !== finalImageUrl) {
                    try {
                        const oldImageRef = ref(storage, editingUser.image);
                        await deleteObject(oldImageRef);
                    } catch (storageErr) {
                        console.error('Failed to clean up old image:', storageErr);
                    }
                }
            }

            const roleLower = newRole.toLowerCase(); // 'admin' or 'investigator'
            const targetCollection = roleLower === 'admin' ? 'admins' : 'investigators';

            // Check if role changed (e.g. from Admin to Investigator)
            const oldRoleLower = editingUser.role.toLowerCase();
            const roleChanged = oldRoleLower !== roleLower;

            const docId = editingUser.id; // Document ID (usually username)

            // Define update data
            const updatedData = {
                name: newName,
                username: editingUser.username,
                nic: newNic,
                image: finalImageUrl,
                role: roleLower,
                status: editingUser.status || 'Active',
                lastLogin: editingUser.lastLogin || 'Never'
            };

            // Only update password if a new one is typed
            if (newPassword.trim()) {
                updatedData.password = newPassword;
            } else {
                // Fetch the existing password from Firestore to preserve it
                const oldCollection = oldRoleLower === 'admin' ? 'admins' : 'investigators';
                const oldDocRef = doc(db, oldCollection, docId);
                const oldDocSnap = await getDoc(oldDocRef);
                if (oldDocSnap.exists()) {
                    updatedData.password = oldDocSnap.data().password;
                } else {
                    updatedData.password = ''; // fallback
                }
            }

            if (roleChanged) {
                // Delete from old collection and write to new collection
                const oldDocRef = doc(db, oldRoleLower === 'admin' ? 'admins' : 'investigators', docId);
                await deleteDoc(oldDocRef);

                const newDocRef = doc(db, targetCollection, docId);
                await setDoc(newDocRef, updatedData);
            } else {
                // Standard update
                const docRef = doc(db, targetCollection, docId);
                await setDoc(docRef, updatedData);
            }

            // Update local state
            setUsers(prev => prev.map(u =>
                u.id === docId ? {
                    ...u,
                    name: newName,
                    role: newRole,
                    nic: newNic,
                    image: finalImageUrl
                } : u
            ));

            // Trigger success animation
            setIsRegisterSuccess(true);

            // Wait 1.8s to show success state before resetting/closing
            setTimeout(() => {
                setNewName('');
                setNewUsername('');
                setNewRole('Investigator');
                setNewNic('');
                setImageFile(null);
                setNewPassword('');
                setEditingUser(null);
                setIsRegisterSuccess(false);
                setShowNewPassword(false);
                setShowAddModal(false);
            }, 1800);

            // Record update in system logs
            addLog('info', `Operator updated: ${editingUser.username}`, `Operator account ${editingUser.username} was updated. Role: ${newRole}.`, 'admin');
        } catch (error) {
            console.error('Error updating operator:', error);
            setFormError(`Failed to update operator: ${error.message || error.toString()}`);
        } finally {
            setIsUploading(false);
        }
    };

    const handleFormSubmit = async (e) => {
        e.preventDefault();
        if (editingUser) {
            await handleUpdateUser();
        } else {
            await handleAddUser();
        }
    };

    // Toggle user status in Firestore
    const toggleStatus = async (id, role, currentStatus, username) => {
        const targetCollection = role.toLowerCase() === 'admin' ? 'admins' : 'investigators';
        const newStatus = currentStatus === 'Active' ? 'Suspended' : 'Active';

        try {
            const docRef = doc(db, targetCollection, id);
            await updateDoc(docRef, { status: newStatus });

            // Update local state
            setUsers(users.map(u =>
                u.id === id ? { ...u, status: newStatus } : u
            ));

            // Record status change in system logs
            addLog('warning', `Operator ${username} status changed to ${newStatus}`, `Account status for operator ${username} was changed from ${currentStatus} to ${newStatus} by administrator.`, 'admin');
        } catch (error) {
            console.error('Error updating status:', error);
            alert(`Failed to update operator status: ${error.message || error.toString()}`);
        }
    };

    // Delete user in Firestore
    const deleteUser = async (id, role, username) => {
        if (window.confirm(`Are you sure you want to remove operator: ${username}?`)) {
            const targetCollection = role.toLowerCase() === 'admin' ? 'admins' : 'investigators';
            try {
                // Find image URL first before deleting the document
                const userObj = users.find(u => u.id === id);
                const imageUrl = userObj?.image;

                const docRef = doc(db, targetCollection, id);
                await deleteDoc(docRef);

                // Update local state
                setUsers(users.filter(u => u.id !== id));

                // If user has an uploaded profile picture, remove it from Storage
                if (imageUrl && imageUrl.includes('firebasestorage.googleapis.com')) {
                    try {
                        const imageRef = ref(storage, imageUrl);
                        await deleteObject(imageRef);
                        console.log('Successfully deleted operator profile image from storage.');
                    } catch (storageErr) {
                        console.error('Failed to delete profile image from storage:', storageErr);
                    }
                }

                // Record deletion in system logs
                addLog('critical', `Operator ${username} removed from system`, `Operator account ${username} (${role}) was permanently deleted from the database.`, 'admin');
            } catch (error) {
                console.error('Error deleting operator:', error);
                alert(`Failed to delete operator from database: ${error.message || error.toString()}`);
            }
        }
    };

    // Filter users
    const filteredUsers = users.filter(u => {
        const term = searchTerm.toLowerCase();
        return u.name.toLowerCase().includes(term) ||
            u.username.toLowerCase().includes(term) ||
            u.role.toLowerCase().includes(term) ||
            u.nic.toLowerCase().includes(term);
    });

    return (
        <div className="user-mgmt-page">
            <AdminHeader />

            <main className="user-mgmt-content">
                <div className="user-mgmt-header-row">
                    <div className="title-group">
                        <h1>Users & Roles</h1>
                    </div>
                    <button className="add-operator-btn" onClick={() => setShowAddModal(true)}>
                        <UserPlus size={18} />
                        <span>Add Operator</span>
                    </button>
                </div>

                <div className="search-controls">
                    <div className="user-search-bar">
                        <Search size={18} color="var(--text-muted)" />
                        <input
                            type="text"
                            placeholder="Search operators by name, email, role or NIC..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                        />
                    </div>
                </div>

                <div className="users-table-container">
                    <table className="users-table">
                        <thead>
                            <tr>
                                <th>Operator Details</th>
                                <th>Credentials (Username)</th>
                                <th>NIC Card</th>
                                <th>Security Role</th>
                                <th>Account Status</th>
                                <th>Last Security Access</th>
                                <th style={{ textAlign: 'center' }}>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {isLoading ? (
                                <tr>
                                    <td colSpan="7" className="no-operators-row">
                                        <div className="loading-spinner-small"></div>
                                        <p style={{ marginTop: '0.5rem' }}>Fetching secure operator database...</p>
                                    </td>
                                </tr>
                            ) : filteredUsers.length === 0 ? (
                                <tr>
                                    <td colSpan="7" className="no-operators-row">
                                        <Search size={30} color="var(--text-muted)" />
                                        <p>No matching operators found.</p>
                                    </td>
                                </tr>
                            ) : (
                                filteredUsers.map((user) => (
                                    <tr
                                        key={user.id}
                                        className={`${user.status === 'Suspended' ? 'suspended-row' : ''} user-row-clickable`}
                                        onClick={() => handleEditClick(user)}
                                        style={{ cursor: 'pointer' }}
                                    >
                                        <td className="op-name-cell">
                                            <div className="op-profile-cell-flex">
                                                <img
                                                    src={user.image || `https://api.dicebear.com/7.x/bottts/svg?seed=${user.id}`}
                                                    alt="Avatar"
                                                    className="op-table-avatar"
                                                />
                                                <span>{user.name}</span>
                                            </div>
                                        </td>
                                        <td className="op-email-cell">{user.username}</td>
                                        <td className="op-nic-cell" style={{ fontFamily: 'monospace' }}>{user.nic || 'N/A'}</td>
                                        <td>
                                            <span className={`role-badge ${user.role.toLowerCase()}`}>
                                                {user.role}
                                            </span>
                                        </td>
                                        <td>
                                            <span className={`status-badge ${user.status.toLowerCase()}`}>
                                                <span className="dot"></span>
                                                {user.status}
                                            </span>
                                        </td>
                                        <td className="op-login-cell">{user.lastLogin}</td>
                                        <td>
                                            <div className="actions-cell">
                                                <button
                                                    className="status-toggle-btn"
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        toggleStatus(user.id, user.role, user.status, user.username);
                                                    }}
                                                    title={user.status === 'Active' ? 'Suspend Operator' : 'Activate Operator'}
                                                >
                                                    {user.status === 'Active' ? (
                                                        <ToggleRight size={22} color="var(--status-found)" />
                                                    ) : (
                                                        <ToggleLeft size={22} color="var(--text-muted)" />
                                                    )}
                                                </button>
                                                <button
                                                    className="delete-operator-btn"
                                                    onClick={(e) => {
                                                        e.stopPropagation();
                                                        deleteUser(user.id, user.role, user.username);
                                                    }}
                                                    title="Delete Operator"
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
            {/* Modal for adding/editing user */}
            {showAddModal && (
                <div
                    className="add-modal-overlay"
                    onClick={(e) => {
                        if (e.target === e.currentTarget) {
                            handleCloseModal();
                        }
                    }}
                >
                    <div className="add-modal-card">
                        {isRegisterSuccess ? (
                            <div className="registration-success-animation">
                                <div className="checkmark-circle">
                                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" width="40" height="40" className="checkmark-svg">
                                        <polyline points="20 6 9 17 4 12" />
                                    </svg>
                                </div>
                                <h3>{editingUser ? 'Operator Updated!' : 'Operator Registered!'}</h3>
                                <p>{editingUser ? 'Local and network data updated successfully.' : 'Local and network credentials compiled successfully.'}</p>
                            </div>
                        ) : (
                            <>
                                <div className="modal-header">
                                    <h2>{editingUser ? 'Edit Operator' : 'Register Operator'}</h2>
                                    <button className="modal-close-btn" onClick={handleCloseModal} disabled={isUploading}>
                                        <X size={20} />
                                    </button>
                                </div>

                                {formError && <div className="form-error-banner">{formError}</div>}

                                <form onSubmit={handleFormSubmit} className="add-operator-form">
                                    <div className="form-field">
                                        <label>Full Name</label>
                                        <input
                                            type="text"
                                            value={newName}
                                            onChange={(e) => setNewName(e.target.value)}
                                            placeholder="e.g. John Doe"
                                            disabled={isUploading}
                                            required
                                        />
                                    </div>

                                    <div className="form-field">
                                        <label>NIC Card Number</label>
                                        <input
                                            type="text"
                                            value={newNic}
                                            onChange={(e) => setNewNic(e.target.value)}
                                            placeholder="e.g. 19990429402"
                                            disabled={isUploading}
                                            required
                                        />
                                    </div>

                                    <div className="form-field">
                                        <label>Security Role</label>
                                        <select
                                            value={newRole}
                                            onChange={(e) => setNewRole(e.target.value)}
                                            disabled={isUploading}
                                        >
                                            <option value="Investigator">Investigator</option>
                                            <option value="Admin">Admin (Full Control)</option>
                                        </select>
                                    </div>

                                    <div className="form-field">
                                        <label>Username</label>
                                        <input
                                            type="text"
                                            value={newUsername}
                                            onChange={(e) => setNewUsername(e.target.value)}
                                            placeholder="e.g. admin_jane or inv_doe"
                                            disabled={isUploading || !!editingUser}
                                            style={editingUser ? { background: 'rgba(255,255,255,0.05)', color: 'var(--text-muted)', cursor: 'not-allowed' } : {}}
                                            required
                                        />
                                        {!editingUser && (
                                            <span className="field-hint">
                                                {newRole === 'Admin' ?
                                                    "Will prefix 'admin_' if not present." :
                                                    "Will prefix 'inv_' if not present."
                                                }
                                            </span>
                                        )}
                                    </div>

                                    <div className="form-field">
                                        <label>Profile Image (Optional)</label>
                                        <div className="profile-image-upload-wrapper">
                                            <input
                                                type="file"
                                                accept="image/*"
                                                style={{ display: 'none' }}
                                                ref={fileInputRef}
                                                onChange={handleImageChange}
                                                disabled={isUploading}
                                            />
                                            <div
                                                className="profile-image-preview-container"
                                                onClick={() => fileInputRef.current.click()}
                                                title="Choose Profile Image"
                                            >
                                                {imageFile ? (
                                                    <>
                                                        <img src={URL.createObjectURL(imageFile)} alt="Profile Preview" />
                                                        <div className="remove-image-badge">Change</div>
                                                    </>
                                                ) : editingUser?.image ? (
                                                    <>
                                                        <img src={editingUser.image} alt="Profile Preview" />
                                                        <div className="remove-image-badge">Change</div>
                                                    </>
                                                ) : (
                                                    <div className="upload-placeholder-content">
                                                        <Camera size={24} className="upload-icon-style" />
                                                        <span className="upload-text-small">Upload</span>
                                                    </div>
                                                )}
                                            </div>
                                            {imageFile && (
                                                <div className="image-file-info">
                                                    <span>{imageFile.name}</span>
                                                    <button
                                                        type="button"
                                                        className="clear-image-btn"
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            setImageFile(null);
                                                        }}
                                                    >
                                                        Clear
                                                    </button>
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                    <div className="form-field">
                                        <label>{editingUser ? 'Access Password (Optional)' : 'Access Password'}</label>
                                        <div className="password-input-wrapper" style={{ position: 'relative' }}>
                                            <input
                                                type={showNewPassword ? "text" : "password"}
                                                value={newPassword}
                                                onChange={(e) => setNewPassword(e.target.value)}
                                                placeholder={editingUser ? "Leave blank to keep unchanged" : "••••••••"}
                                                disabled={isUploading}
                                                style={{ width: '100%', paddingRight: '2.5rem' }}
                                                required={!editingUser}
                                            />
                                            <button
                                                type="button"
                                                className="password-toggle-btn"
                                                onClick={() => setShowNewPassword(!showNewPassword)}
                                                style={{
                                                    position: 'absolute',
                                                    right: '10px',
                                                    top: '50%',
                                                    transform: 'translateY(-50%)',
                                                    background: 'transparent',
                                                    border: 'none',
                                                    color: 'var(--text-muted)',
                                                    cursor: 'pointer',
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    justifyContent: 'center',
                                                    padding: '4px'
                                                }}
                                            >
                                                {showNewPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                                            </button>
                                        </div>
                                        <span className="field-hint" style={{ color: 'var(--text-muted)', fontSize: '0.72rem' }}>
                                            {editingUser ?
                                                "Only fill this in if you want to change the operator's password. Must be at least 8 characters with uppercase, lowercase, number, and special character." :
                                                "Must be at least 8 characters and include uppercase, lowercase, number, and special character."
                                            }
                                        </span>
                                    </div>

                                    <button type="submit" className="form-submit-btn" disabled={isUploading}>
                                        {isUploading ? 'Securing Operator...' : editingUser ? 'Update Operator' : 'Save to Database'}
                                    </button>
                                </form>
                            </>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
};

export default UserManagement;
