import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { ArrowLeft, Server, UserCheck, Eye, EyeOff, ShieldAlert } from 'lucide-react';
import logo from '../assets/logo.png';
import { db } from '../firebaseConfig';
import { collection, query, where, getDocs } from 'firebase/firestore';
import './Login.css';

const Login = () => {
    const navigate = useNavigate();
    const { login } = useAuth();
    const [step, setStep] = useState('role-select');
    const [selectedRole, setSelectedRole] = useState('');
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [loading, setLoading] = useState(false);
    const [isSuccess, setIsSuccess] = useState(false);
    const [rootAdminEmails, setRootAdminEmails] = useState(['admin', 'rootadmin@argus.com']);
    const [activeAdminEmails, setActiveAdminEmails] = useState([]);

    useEffect(() => {
        const fetchAdminEmails = async () => {
            try {
                // 1. Fetch Root Admins specifically
                const qRoot = query(collection(db, 'admins'), where('role', '==', 'Root Admin'));
                const rootSnapshot = await getDocs(qRoot);
                const roots = [];
                rootSnapshot.forEach((doc) => {
                    const data = doc.data();
                    if (data.email) {
                        roots.push(data.email);
                    } else if (data.username) {
                        roots.push(data.username);
                    }
                });
                setRootAdminEmails(roots.length > 0 ? roots : ['admin', 'rootadmin@argus.com']);

                // 2. Fetch all active admin emails
                const qActive = query(collection(db, 'admins'), where('status', '==', 'Active'));
                const activeSnapshot = await getDocs(qActive);
                const activeEmails = [];
                activeSnapshot.forEach((doc) => {
                    const data = doc.data();
                    if (data.email) {
                        activeEmails.push(data.email);
                    } else if (data.username) {
                        activeEmails.push(data.username);
                    }
                });
                setActiveAdminEmails(activeEmails.length > 0 ? activeEmails : ['admin', 'rootadmin@argus.com']);
            } catch (err) {
                console.error('Error fetching admin emails:', err);
                setRootAdminEmails(['admin', 'rootadmin@argus.com']);
                setActiveAdminEmails(['admin', 'rootadmin@argus.com']);
            }
        };
        fetchAdminEmails();
    }, []);

    const handleRoleSelect = (role) => {
        setSelectedRole(role);
        setStep('credentials');
        setError('');
        setShowPassword(false);
    };

    const handleBackToRoles = () => {
        setStep('role-select');
        setSelectedRole('');
        setError('');
        setShowPassword(false);
    };

    const handleLogin = async (e) => {
        e.preventDefault();

        try {
            setError('');
            setLoading(true);

            const loggedUser = await login(username, password, selectedRole);

            // Trigger button checkmark and card exit states
            setIsSuccess(true);
            setLoading(false);

            // Redirect after 1.2s to show success feedback animation
            setTimeout(() => {
                const roleLower = (loggedUser.role || '').toLowerCase();
                if (roleLower === 'admin' || roleLower === 'root admin') {
                    navigate('/admin/dashboard');
                } else {
                    navigate('/dashboard');
                }
            }, 1200);
        } catch (err) {
            setError(err.message || 'Failed to log in. Please check your credentials.');
            console.error(err);
            setLoading(false);
        }
    };

    return (
        <div className="login-container">
            <div className="login-left">
                <div className="brand-wrapper">
                    <img src={logo} alt="Argus Logo" className="huge-logo" />
                    <h1 className="brand-title-text">ARGUS</h1>
                    <p className="slogan-text">See. Know. Secure</p>
                </div>
            </div>

            <div className="login-right">
                <div className={`form-wrapper ${isSuccess ? 'success-exit' : ''}`}>
                    {step === 'role-select' ? (
                        <div className="role-selection-wrapper">
                            <h1 className="sign-in-title">Access Gate</h1>
                            <p className="sign-in-subtitle">Select Your Role to Login</p>

                            <div className="role-cards-container">
                                <div className="role-select-card admin" onClick={() => handleRoleSelect('admin')}>
                                    <div className="role-card-icon-wrapper">
                                        <Server size={32} />
                                    </div>
                                    <div className="role-card-info">
                                        <h2>Administrator</h2>
                                    </div>
                                    <div className="select-badge">ADMIN PANEL</div>
                                </div>

                                <div className="role-select-card investigator" onClick={() => handleRoleSelect('investigator')}>
                                    <div className="role-card-icon-wrapper">
                                        <UserCheck size={32} />
                                    </div>
                                    <div className="role-card-info">
                                        <h2>Investigator</h2>
                                    </div>
                                    <div className="select-badge">INVESTIGATOR PANEL</div>
                                </div>
                            </div>
                        </div>
                    ) : step === 'credentials' ? (
                        <div className="credentials-form-wrapper">
                            <button className="back-roles-btn" onClick={handleBackToRoles} disabled={isSuccess}>
                                <ArrowLeft size={16} />
                                <span>Switch Role Access</span>
                            </button>

                            <div className="credentials-header">
                                <h1 className="sign-in-title">Authenticate</h1>
                                <p className="sign-in-subtitle">
                                    Signing in as <span className="active-role-text">{selectedRole === 'admin' ? 'Administrator' : 'Investigator'}</span>
                                </p>
                            </div>

                            {error && <div className="error-message">{error}</div>}

                            <form onSubmit={handleLogin} className="signin-form">
                                <div className="form-group">
                                    <label className="input-label">Username</label>
                                    <input
                                        type="text"
                                        className="styled-input"
                                        value={username}
                                        onChange={(e) => setUsername(e.target.value)}
                                        placeholder="enter your username"
                                        disabled={isSuccess}
                                        required
                                    />
                                </div>

                                <div className="form-group">
                                    <label className="input-label">Password</label>
                                    <div className="password-input-wrapper" style={{ position: 'relative' }}>
                                        <input
                                            type={showPassword ? "text" : "password"}
                                            className="styled-input"
                                            value={password}
                                            onChange={(e) => setPassword(e.target.value)}
                                            placeholder="enter your password"
                                            disabled={isSuccess}
                                            style={{ paddingRight: '2.5rem' }}
                                            required
                                        />
                                        <button
                                            type="button"
                                            className="password-toggle-btn"
                                            onClick={() => setShowPassword(!showPassword)}
                                            style={{
                                                position: 'absolute',
                                                right: '12px',
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
                                            {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                                        </button>
                                    </div>
                                </div>

                                <div className="forgot-password-link">
                                    <a href="#" onClick={(e) => { e.preventDefault(); setStep('forgot-password'); }}>Forget Password ?</a>
                                </div>

                                <button type="submit" className={`login-btn ${isSuccess ? 'success' : ''}`} disabled={loading || isSuccess}>
                                    {isSuccess ? (
                                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" width="20" height="20" className="btn-success-check-svg">
                                            <polyline points="20 6 9 17 4 12" />
                                        </svg>
                                    ) : loading ? (
                                        'Validating Credentials...'
                                    ) : (
                                        'Login'
                                    )}
                                </button>
                            </form>
                        </div>
                    ) : (
                        <div className="forgot-password-wrapper">
                            <button className="back-roles-btn" onClick={() => setStep('credentials')}>
                                <ArrowLeft size={16} />
                                <span>Back to Login</span>
                            </button>

                            <div className="forgot-password-card">
                                <div className="forgot-password-icon-wrapper">
                                    <ShieldAlert size={32} />
                                </div>
                                <h1 className="forgot-password-title">Access Recovery</h1>
                                <p className="forgot-password-message">
                                    To Reset Your Password, Please Contact System Administrator
                                </p>
                                <div className="forgot-password-admin-info">
                                    <p className="forgot-password-admin-text">
                                        Admin email : <span className="forgot-password-email-value">
                                            {selectedRole === 'admin' ? rootAdminEmails.join(', ') : activeAdminEmails.join(', ')}
                                        </span>
                                    </p>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default Login;
