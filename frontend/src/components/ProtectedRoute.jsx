import React, { useEffect } from 'react';
import { Navigate, useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';

const ProtectedRoute = ({ children, allowedRole }) => {
    const { currentUser, logout } = useAuth();
    const navigate = useNavigate();

    // Note: Operator active status, suspension, and revocation are authoritatively tracked
    // in real-time by AuthContext's onSnapshot listener. Eliminating redundant blocking Firestore getDocs
    // here ensures instant route transitions without weakening security.
    useEffect(() => {
        if (currentUser && currentUser.status === 'Suspended') {
            logout();
            navigate('/');
        }
    }, [currentUser, navigate, logout]);

    if (!currentUser) {
        return <Navigate to="/" />;
    }

    const userRole = (currentUser.role || '').toLowerCase();
    
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
