import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { ArrowLeft, Server, UserCheck } from 'lucide-react';
import logo from '../assets/logo.png';
import './Login.css';

const Login = () => {
    const navigate = useNavigate();
    const { login } = useAuth();
    const [step, setStep] = useState('role-select');
    const [selectedRole, setSelectedRole] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const [isSuccess, setIsSuccess] = useState(false);

    const handleRoleSelect = (role) => {
        setSelectedRole(role);
        setStep('credentials');
        setError('');
    };

    const handleBackToRoles = () => {
        setStep('role-select');
        setSelectedRole('');
        setError('');
    };

    const handleLogin = async (e) => {
        e.preventDefault();

        try {
            setError('');
            setLoading(true);
            
            // Format username to email if it does not contain '@'
            let loginEmail = email;
            if (!email.includes('@')) {
                loginEmail = `${email}@argus.com`;
            }
            
            const loggedUser = await login(loginEmail, password, selectedRole);
            
            // Trigger button checkmark and card exit states
            setIsSuccess(true);
            setLoading(false);

            // Redirect after 1.2s to show success feedback animation
            setTimeout(() => {
                if (loggedUser.role === 'admin') {
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
                                        <h2>System Administrator</h2>
                                    </div>
                                    <div className="select-badge">ADMIN\'S VIEW</div>
                                </div>

                                <div className="role-select-card investigator" onClick={() => handleRoleSelect('investigator')}>
                                    <div className="role-card-icon-wrapper">
                                        <UserCheck size={32} />
                                    </div>
                                    <div className="role-card-info">
                                        <h2>Field Investigator</h2>
                                    </div>
                                    <div className="select-badge">INVESTIGATOR\'S VIEW</div>
                                </div>
                            </div>
                        </div>
                    ) : (
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
                                        value={email}
                                        onChange={(e) => setEmail(e.target.value)}
                                        placeholder="enter your username"
                                        disabled={isSuccess}
                                        required
                                    />
                                </div>

                                <div className="form-group">
                                    <label className="input-label">Password</label>
                                    <input
                                        type="password"
                                        className="styled-input"
                                        value={password}
                                        onChange={(e) => setPassword(e.target.value)}
                                        placeholder="enter your password"
                                        disabled={isSuccess}
                                        required
                                    />
                                </div>

                                <div className="forgot-password-link">
                                    <a href="#" onClick={(e) => e.preventDefault()}>Forget Password ?</a>
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
                    )}
                </div>
            </div>
        </div>
    );
};

export default Login;
