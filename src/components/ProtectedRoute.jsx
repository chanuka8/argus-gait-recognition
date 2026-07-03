import React, { useEffect } from 'react';
import { Navigate, useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { db } from '../firebaseConfig';
import { collection, query, where, getDocs } from 'firebase/firestore';

const ProtectedRoute = ({ children, allowedRole }) => {
    const { currentUser, logout } = useAuth();
    const navigate = useNavigate();

    useEffect(() => {
        const verifyActiveStatus = async () => {
            if (!currentUser) return;
            try {
                const roleLower = currentUser.role.toLowerCase();
                const targetCollection = (roleLower === 'admin' || roleLower === 'root admin') ? 'admins' : 'investigators';
                const userQuery = query(
                    collection(db, targetCollection),
                    where('username', '==', currentUser.username)
                );
                const querySnapshot = await getDocs(userQuery);
                if (!querySnapshot.empty) {
                    const userData = querySnapshot.docs[0].data();
                    if (userData.status === 'Suspended') {
                        alert('Your account has been suspended. Logging out.');
                        await logout();
                        navigate('/');
                    }
                } else {
                    alert('Operator account no longer exists. Logging out.');
                    await logout();
                    navigate('/');
                }
            } catch (err) {
                console.error("Error verifying active status:", err);
            }
        };

        verifyActiveStatus();
    }, [currentUser, navigate, logout]);

    if (!currentUser) {
        return <Navigate to="/" />;
    }

    const userRole = (currentUser.role || '').toLowerCase();
    
    // Strict isolation checks based on session role
    if (allowedRole === 'admin' && userRole !== 'admin' && userRole !== 'root admin') {
        if (userRole === 'investigator') {
            return <Navigate to="/dashboard" />;
        }
        return <Navigate to="/" />;
    }

    if (allowedRole === 'investigator' && userRole !== 'investigator') {
        if (userRole === 'admin' || userRole === 'root admin') {
            return <Navigate to="/admin/dashboard" />;
        }
        return <Navigate to="/" />;
    }

    return children;
};

export default ProtectedRoute;
