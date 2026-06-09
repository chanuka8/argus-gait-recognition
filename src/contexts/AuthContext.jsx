import React, { createContext, useContext, useState, useEffect } from 'react';
import { db } from '../firebaseConfig';
import { collection, query, where, getDocs, doc, setDoc } from 'firebase/firestore';

const AuthContext = createContext();

export const useAuth = () => {
    return useContext(AuthContext);
};

export const AuthProvider = ({ children }) => {
    const [currentUser, setCurrentUser] = useState(null);
    const [loading, setLoading] = useState(true);

    // 1. Seed database with default admin & investigator if they don't exist
    const seedDatabase = async () => {
        try {
            // Check admins collection
            const adminQuery = query(collection(db, 'admins'), where('username', '==', 'admin@argus.com'));
            const adminSnapshot = await getDocs(adminQuery);
            if (adminSnapshot.empty) {
                const newAdminRef = doc(collection(db, 'admins'), 'admin_root');
                await setDoc(newAdminRef, {
                    name: 'Root Administrator',
                    username: 'admin@argus.com',
                    password: 'admin123',
                    nic: '199502104523',
                    image: 'https://api.dicebear.com/7.x/bottts/svg?seed=admin',
                    role: 'admin',
                    status: 'Active',
                    lastLogin: 'Never'
                });
                console.log('Seeded root admin account.');
            }

            // Check investigators collection
            const invQuery = query(collection(db, 'investigators'), where('username', '==', 'inv_john@argus.com'));
            const invSnapshot = await getDocs(invQuery);
            if (invSnapshot.empty) {
                const newInvRef = doc(collection(db, 'investigators'), 'inv_john');
                await setDoc(newInvRef, {
                    name: 'Senior Investigator',
                    username: 'inv_john@argus.com',
                    password: 'inv123',
                    nic: '199201402941',
                    image: 'https://api.dicebear.com/7.x/bottts/svg?seed=john',
                    role: 'investigator',
                    status: 'Active',
                    lastLogin: 'Never'
                });
                console.log('Seeded senior investigator account.');
            }
        } catch (error) {
            console.error('Error seeding database:', error);
        }
    };

    // Initialize session and seed database on mount
    useEffect(() => {
        const checkSessionAndSeed = async () => {
            await seedDatabase();
            
            const cachedUser = localStorage.getItem('argus_current_user');
            if (cachedUser) {
                setCurrentUser(JSON.parse(cachedUser));
            }
            setLoading(false);
        };
        
        checkSessionAndSeed();
    }, []);

    // 2. Custom Login Logic querying Firestore
    const login = async (username, password, role) => {
        if (!role) {
            throw new Error('System access role selection is required.');
        }

        // Format username to email if no '@' exists
        let formattedUsername = username.trim().toLowerCase();
        if (!formattedUsername.includes('@')) {
            formattedUsername = `${formattedUsername}@argus.com`;
        }

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
        localStorage.setItem('argus_current_user', JSON.stringify(loggedUser));
        setCurrentUser(loggedUser);
        return loggedUser;
    };

    // 3. Logout logic
    const logout = async () => {
        localStorage.removeItem('argus_current_user');
        setCurrentUser(null);
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
