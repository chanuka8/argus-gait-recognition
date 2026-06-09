import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

const ProtectedRoute = ({ children, allowedRole }) => {
    const { currentUser } = useAuth();

    if (!currentUser) {
        return <Navigate to="/" />;
    }

    const userRole = currentUser.role || '';
    
    // Strict isolation checks based on session role
    if (allowedRole === 'admin' && userRole !== 'admin') {
        if (userRole === 'investigator') {
            return <Navigate to="/dashboard" />;
        }
        return <Navigate to="/" />;
    }

    if (allowedRole === 'investigator' && userRole !== 'investigator') {
        if (userRole === 'admin') {
            return <Navigate to="/admin/dashboard" />;
        }
        return <Navigate to="/" />;
    }

    return children;
};

export default ProtectedRoute;
