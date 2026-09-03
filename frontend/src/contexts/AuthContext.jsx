import React, { useState, useEffect } from 'react';
import { db } from '../firebaseConfig';
import { collection, query, where, onSnapshot } from 'firebase/firestore';
import { addLog } from '../utils/logService';
import { AuthContext } from './authContextDef';
import { API_BASE } from '../config/apiConfig';

export const AuthProvider = ({ children }) => {
    const [currentUser, setCurrentUser] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const checkSession = async () => {
            const token = sessionStorage.getItem('argus_session_token');
            const cachedUser = sessionStorage.getItem('argus_current_user');

            if (token) {
                try {
                    const res = await fetch(`${API_BASE}/api/v1/auth/me`, {
                        headers: {
                            'Authorization': `Bearer ${token}`,
                        },
                    });

                    if (res.ok) {
                        const profile = await res.json();
                        sessionStorage.setItem('argus_current_user', JSON.stringify(profile));
                        setCurrentUser(profile);
                    } else {
                        // Session expired or invalid on server
                        sessionStorage.removeItem('argus_session_token');
                        sessionStorage.removeItem('argus_current_user');
                        setCurrentUser(null);
                    }
                } catch {
                    // Fallback to cached profile if network temporarily unavailable
                    if (cachedUser) {
                        try {
                            setCurrentUser(JSON.parse(cachedUser));
                        } catch {
                            setCurrentUser(null);
                        }
                    } else {
                        setCurrentUser(null);
                    }
                }
            } else {
                sessionStorage.removeItem('argus_session_token');
                sessionStorage.removeItem('argus_current_user');
                setCurrentUser(null);
            }
            setLoading(false);
        };

        checkSession();
    }, []);

    // Listen for operator suspension or removal events in real-time
    useEffect(() => {
        if (!currentUser || !currentUser.username) return;

        const roleLower = (currentUser.role || '').toLowerCase();
        const targetCollection = (roleLower === 'admin' || roleLower === 'root admin' || roleLower === 'root_admin')
            ? 'admins'
            : 'investigators';

        const q = query(
            collection(db, targetCollection),
            where('username', '==', currentUser.username.toLowerCase())
        );

        const unsubscribe = onSnapshot(q, (querySnapshot) => {
            if (!querySnapshot.empty) {
                const data = querySnapshot.docs[0].data();
                if (data.status === 'Suspended') {
                    alert('Your account has been suspended. Logging out.');
                    sessionStorage.removeItem('argus_session_token');
                    sessionStorage.removeItem('argus_current_user');
                    setCurrentUser(null);
                    addLog('info', `Operator ${currentUser.username} suspended and logged out`, 'Session terminated automatically due to administrator suspension.', currentUser.username);
                }
            } else {
                // If operator document was deleted
                sessionStorage.removeItem('argus_session_token');
                sessionStorage.removeItem('argus_current_user');
                setCurrentUser(null);
                addLog('info', `Operator ${currentUser.username} removed and logged out`, 'Session terminated automatically because the operator account was deleted.', currentUser.username);
            }
        }, (error) => {
            console.error("Error listening to user status changes:", error);
        });

        return () => unsubscribe();
    }, [currentUser]);

    const login = async (username, password, role) => {
        if (!role) {
            throw new Error('System access role selection is required.');
        }

        const formattedUsername = username.trim().toLowerCase();

        // Authoritative server-side authentication with Argon2id verification
        const res = await fetch(`${API_BASE}/api/v1/auth/login`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                username: formattedUsername,
                password: password,
                role: role,
            }),
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || 'Invalid credentials or login failure.');
        }

        const authData = await res.json();
        const loggedUser = authData.operator;

        sessionStorage.setItem('argus_session_token', authData.token);
        sessionStorage.setItem('argus_current_user', JSON.stringify(loggedUser));
        setCurrentUser(loggedUser);

        addLog(
            'info',
            `Operator ${loggedUser.username} logged in successfully`,
            `User ${loggedUser.name || loggedUser.username} (${loggedUser.role}) authenticated via server-side session token. Session started.`,
            loggedUser.username
        );

        return loggedUser;
    };

    const logout = async () => {
        const token = sessionStorage.getItem('argus_session_token');
        const username = currentUser?.username || 'unknown';
        const name = currentUser?.name || 'Unknown';

        if (token) {
            try {
                await fetch(`${API_BASE}/api/v1/auth/logout`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${token}`,
                    },
                });
            } catch (err) {
                console.warn("Backend session revocation notice:", err);
            }
        }

        sessionStorage.removeItem('argus_session_token');
        sessionStorage.removeItem('argus_current_user');
        setCurrentUser(null);

        addLog('info', `Operator ${username} logged out`, `User ${name} ended their session and was signed out of the system.`, username);
    };

    const value = {
        currentUser,
        login,
        logout,
    };

    return (
        <AuthContext.Provider value={value}>
            {!loading && children}
        </AuthContext.Provider>
    );
};

export default AuthProvider;
