import React, { useState, useEffect } from 'react';
import { UserPlus, Search, ToggleLeft, ToggleRight, Trash2, X } from 'lucide-react';
import AdminHeader from './AdminHeader';
import { db } from '../firebaseConfig';
import { collection, getDocs, doc, setDoc, deleteDoc, updateDoc } from 'firebase/firestore';
import { addLog } from '../utils/logService';
import './UserManagement.css';

const UserManagement = () => {
    const [users, setUsers] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');
    const [showAddModal, setShowAddModal] = useState(false);
    
    // Add user form states
    const [newName, setNewName] = useState('');
    const [newUsername, setNewUsername] = useState('');
    const [newRole, setNewRole] = useState('Investigator');
    const [newNic, setNewNic] = useState('');
    const [newImageSeed, setNewImageSeed] = useState('');
    const [newPassword, setNewPassword] = useState('');
    const [formError, setFormError] = useState('');

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
                    email: data.username,
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
                    email: data.username,
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

        // Validate that username contains role identifiers
        let finalUsername = newUsername.trim().toLowerCase();
        if (newRole === 'Admin' && !finalUsername.includes('admin')) {
            finalUsername = `admin_${finalUsername}`;
        } else if (newRole === 'Investigator' && !finalUsername.includes('inv')) {
            finalUsername = `inv_${finalUsername}`;
        }

        // Add domain if not present
        if (!finalUsername.includes('@')) {
            finalUsername = `${finalUsername}@argus.com`;
        }

        // Check duplicates locally first
        if (users.some(u => u.email.toLowerCase() === finalUsername.toLowerCase())) {
            setFormError('Operator with this username/email already exists.');
            return;
        }

        // Generate Avatar URL
        let finalImageUrl = newImageSeed.trim();
        if (!finalImageUrl) {
            finalImageUrl = `https://api.dicebear.com/7.x/bottts/svg?seed=${finalUsername.split('@')[0]}`;
        } else if (!finalImageUrl.startsWith('http')) {
            finalImageUrl = `https://api.dicebear.com/7.x/bottts/svg?seed=${finalImageUrl}`;
        }

        const roleLower = newRole.toLowerCase(); // 'admin' or 'investigator'
        const targetCollection = roleLower === 'admin' ? 'admins' : 'investigators';
        const docId = finalUsername.split('@')[0];

        try {
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
                    email: finalUsername,
                    role: newRole,
                    nic: newNic,
                    image: finalImageUrl,
                    status: 'Active',
                    lastLogin: 'Never'
                }
            ]);

            // Reset form
            setNewName('');
            setNewUsername('');
            setNewRole('Investigator');
            setNewNic('');
            setNewImageSeed('');
            setNewPassword('');
            setShowAddModal(false);

            // Record user creation in system logs
            addLog('info', `New operator registered: ${finalUsername}`, `Operator ${newName} was created with role ${newRole}. NIC: ${newNic}. Account status: Active.`, 'admin');
        } catch (error) {
            console.error('Error creating operator:', error);
            setFormError('Failed to save operator to database.');
        }
    };

    // Toggle user status in Firestore
    const toggleStatus = async (id, role, currentStatus, email) => {
        if (email === 'admin@argus.com') {
            alert('Cannot suspend root admin.');
            return;
        }

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
            addLog('warning', `Operator ${email} status changed to ${newStatus}`, `Account status for operator ${email} was changed from ${currentStatus} to ${newStatus} by administrator.`, 'admin');
        } catch (error) {
            console.error('Error updating status:', error);
            alert('Failed to update operator status.');
        }
    };

    // Delete user in Firestore
    const deleteUser = async (id, role, email) => {
        if (email === 'admin@argus.com') {
            alert('Cannot delete root admin.');
            return;
        }

        if (window.confirm(`Are you sure you want to remove operator: ${email}?`)) {
            const targetCollection = role.toLowerCase() === 'admin' ? 'admins' : 'investigators';
            try {
                const docRef = doc(db, targetCollection, id);
                await deleteDoc(docRef);

                // Update local state
                setUsers(users.filter(u => u.id !== id));

                // Record deletion in system logs
                addLog('critical', `Operator ${email} removed from system`, `Operator account ${email} (${role}) was permanently deleted from the database.`, 'admin');
            } catch (error) {
                console.error('Error deleting operator:', error);
                alert('Failed to delete operator from database.');
            }
        }
    };

    // Filter users
    const filteredUsers = users.filter(u => {
        const term = searchTerm.toLowerCase();
        return u.name.toLowerCase().includes(term) || 
               u.email.toLowerCase().includes(term) ||
               u.role.toLowerCase().includes(term) ||
               u.nic.toLowerCase().includes(term);
    });

    return (
        <div className="user-mgmt-page">
            <AdminHeader />
            
            <main className="user-mgmt-content">
                <div className="user-mgmt-header-row">
                    <div className="title-group">
                        <h1>User & Role Management</h1>
                        <p>Configure operator databases, credential validation, and authentication parameters.</p>
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
                                <th>Credentials (Email)</th>
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
                                    <tr key={user.id} className={user.status === 'Suspended' ? 'suspended-row' : ''}>
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
                                        <td className="op-email-cell">{user.email}</td>
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
                                                    onClick={() => toggleStatus(user.id, user.role, user.status, user.email)}
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
                                                    onClick={() => deleteUser(user.id, user.role, user.email)}
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

            {/* Modal for adding user */}
            {showAddModal && (
                <div className="add-modal-overlay" onClick={() => setShowAddModal(false)}>
                    <div className="add-modal-card" onClick={(e) => e.stopPropagation()}>
                        <div className="modal-header">
                            <h2>Register Operator</h2>
                            <button className="modal-close-btn" onClick={() => setShowAddModal(false)}>
                                <X size={20} />
                            </button>
                        </div>

                        {formError && <div className="form-error-banner">{formError}</div>}

                        <form onSubmit={handleAddUser} className="add-operator-form">
                            <div className="form-field">
                                <label>Full Name</label>
                                <input 
                                    type="text" 
                                    value={newName} 
                                    onChange={(e) => setNewName(e.target.value)} 
                                    placeholder="e.g. John Doe"
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
                                    required 
                                />
                            </div>

                            <div className="form-field">
                                <label>Security Role</label>
                                <select 
                                    value={newRole} 
                                    onChange={(e) => setNewRole(e.target.value)}
                                >
                                    <option value="Investigator">Investigator</option>
                                    <option value="Admin">Admin (Full Control)</option>
                                </select>
                            </div>

                            <div className="form-field">
                                <label>Username / Email</label>
                                <input 
                                    type="text" 
                                    value={newUsername} 
                                    onChange={(e) => setNewUsername(e.target.value)} 
                                    placeholder="Username (contains 'admin' or 'inv')"
                                    required 
                                />
                                <span className="field-hint">
                                    {newRole === 'Admin' ? 
                                        "Will append 'admin' automatically if not present." : 
                                        "Will append 'inv' automatically if not present."
                                    }
                                </span>
                            </div>

                            <div className="form-field">
                                <label>Avatar Seed or Image URL</label>
                                <input 
                                    type="text" 
                                    value={newImageSeed} 
                                    onChange={(e) => setNewImageSeed(e.target.value)} 
                                    placeholder="e.g. agent7 or http://image-url"
                                />
                                <span className="field-hint">Used to generate your operator avatar thumbnail.</span>
                            </div>

                            <div className="form-field">
                                <label>Access Password</label>
                                <input 
                                    type="password" 
                                    value={newPassword} 
                                    onChange={(e) => setNewPassword(e.target.value)} 
                                    placeholder="••••••••"
                                    required 
                                />
                            </div>

                            <button type="submit" className="form-submit-btn">
                                Save to Database
                            </button>
                        </form>
                    </div>
                </div>
            )}
        </div>
    );
};

export default UserManagement;
