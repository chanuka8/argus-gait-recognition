/* eslint-disable react-refresh/only-export-components */
import React, { createContext, useContext, useState, useEffect } from 'react';
import { db } from '../firebaseConfig';
import { collection, query, where, getDocs, doc, setDoc, onSnapshot } from 'firebase/firestore';
import { addLog } from '../utils/logService';

const AuthContext = createContext();

export const useAuth = () => {
    return useContext(AuthContext);
};

export const AuthProvider = ({ children }) => {
    const [currentUser, setCurrentUser] = useState(null);
    const [loading, setLoading] = useState(true);

    // 1. Seed database with default admin & investigator if collections are empty
    const seedDatabase = async () => {
        try {
            // Check admins collection
            const adminSnapshot = await getDocs(collection(db, 'admins'));
            if (adminSnapshot.empty) {
                const newAdminRef = doc(collection(db, 'admins'), 'admin_root');
                await setDoc(newAdminRef, {
                    name: 'Root Administrator',
                    username: 'admin',
                    password: 'Admin@123',
                    nic: '199502104523',
                    image: 'https://api.dicebear.com/7.x/bottts/svg?seed=admin',
                    role: 'admin',
                    status: 'Active',
                    lastLogin: 'Never'
                });
                console.log('Seeded root admin account.');
            }

        } catch (error) {
            console.error('Error seeding database:', error);
        }
    };

    // Initialize session and seed database on mount
    useEffect(() => {
        const checkSessionAndSeed = async () => {
            await seedDatabase();
            
            const cachedUser = sessionStorage.getItem('argus_current_user');
            if (cachedUser) {
                const parsedUser = JSON.parse(cachedUser);
                try {
                    const roleLower = parsedUser.role.toLowerCase();
                    const targetCollection = roleLower === 'admin' ? 'admins' : 'investigators';
                    const userQuery = query(
                        collection(db, targetCollection),
                        where('username', '==', parsedUser.username)
                    );
                    const querySnapshot = await getDocs(userQuery);
                    if (!querySnapshot.empty) {
                        const userData = querySnapshot.docs[0].data();
                        if (userData.status === 'Suspended') {
                            console.warn('Current user is suspended. Logging out.');
                            sessionStorage.removeItem('argus_current_user');
                            setCurrentUser(null);
                        } else {
                            // Update cached information with live data
                            const updatedUser = {
                                ...parsedUser,
                                id: querySnapshot.docs[0].id,
                                name: userData.name,
                                image: userData.image,
                                nic: userData.nic,
                                role: userData.role || roleLower,
                                status: userData.status
                            };
                            sessionStorage.setItem('argus_current_user', JSON.stringify(updatedUser));
                            setCurrentUser(updatedUser);
                        }
                    } else {
                        // User no longer exists in Firestore
                        sessionStorage.removeItem('argus_current_user');
                        setCurrentUser(null);
                    }
                } catch (err) {
                    console.error('Error verifying user session:', err);
                    // Fallback to cache if offline
                    setCurrentUser(parsedUser);
                }
            }
            setLoading(false);
        };
        
        checkSessionAndSeed();
    }, []);

    useEffect(() => {
        if (!currentUser) return;

        const roleLower = currentUser.role.toLowerCase();
        const targetCollection = roleLower === 'admin' ? 'admins' : 'investigators';
        const q = query(
            collection(db, targetCollection),
            where('username', '==', currentUser.username)
        );

        const unsubscribe = onSnapshot(q, (querySnapshot) => {
            if (!querySnapshot.empty) {
                const data = querySnapshot.docs[0].data();
                if (data.status === 'Suspended') {
                    alert('Your account has been suspended. Logging out.');
                    sessionStorage.removeItem('argus_current_user');
                    setCurrentUser(null);
                    addLog('info', `Operator ${currentUser.username} suspended and logged out`, `Session terminated automatically due to administrator suspension.`, currentUser.username);
                }
            } else {
                alert('Your account has been removed. Logging out.');
                sessionStorage.removeItem('argus_current_user');
                setCurrentUser(null);
                addLog('info', `Operator ${currentUser.username} removed and logged out`, `Session terminated automatically because the operator account was deleted.`, currentUser.username);
            }
        }, (error) => {
            console.error("Error listening to user status changes:", error);
        });

        return () => unsubscribe();
    }, [currentUser]);

    // 2. Custom Login Logic querying Firestore
    const login = async (username, password, role) => {
        if (!role) {
            throw new Error('System access role selection is required.');
        }

        // Use clean username
        const formattedUsername = username.trim().toLowerCase();

        const roleLower = role.toLowerCase();
        const targetCollection = roleLower === 'admin' ? 'admins' : 'investigators';
        const userQuery = query(
            collection(db, targetCollection),
            where('username', '==', formattedUsername)
        );

        const querySnapshot = await getDocs(userQuery);
        
        if (querySnapshot.empty) {
            throw new Error('Operator account not found.');
        }

        // Get matching document
        const userDoc = querySnapshot.docs[0];
        const userData = userDoc.data();

        // Enforce account status check
        if (userData.status === 'Suspended') {
            throw new Error('Account suspended. Contact administration.');
        }

        // Enforce password check
        if (userData.password !== password) {
            throw new Error('Invalid credentials.');
        }

        // Session data to cache
        const loggedUser = {
            id: userDoc.id,
            name: userData.name,
            email: userData.username, // mapping for route guard compatibilities
            username: userData.username,
            nic: userData.nic,
            image: userData.image,
            role: userData.role || roleLower
        };

        // Cache and set state
        sessionStorage.setItem('argus_current_user', JSON.stringify(loggedUser));
        setCurrentUser(loggedUser);

        // Record login event in system logs
        addLog('info', `Operator ${loggedUser.username} logged in successfully`, `User ${loggedUser.name} (${loggedUser.role}) authenticated via credential verification. Session started.`, loggedUser.username);

        return loggedUser;
    };

    // 3. Logout logic
    const logout = async () => {
        const username = currentUser?.username || 'unknown';
        const name = currentUser?.name || 'Unknown';

        sessionStorage.removeItem('argus_current_user');
        setCurrentUser(null);

        // Record logout event in system logs
        addLog('info', `Operator ${username} logged out`, `User ${name} ended their session and was signed out of the system.`, username);
    };

    const value = {
        currentUser,
        login,
        logout
    };

    return (
        <AuthContext.Provider value={value}>
            {!loading && children}
        </AuthContext.Provider>
    );
};
