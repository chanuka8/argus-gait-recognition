import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import logo from '../assets/logo.png';
import './Login.css';

const Login = () => {
    const navigate = useNavigate();
    const { login } = useAuth();
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);

    const handleLogin = async (e) => {
        e.preventDefault();

        try {
            setError('');
            setLoading(true);
            await login(email, password);
            navigate('/dashboard');
        } catch (err) {
            setError('Failed to log in. Please check your credentials.');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="login-container">
            <div className="login-left">
                <img src={logo} alt="Argus Logo" className="huge-logo" />
            </div>

            <div className="login-right">
                <div className="form-wrapper">
                    <h1 className="sign-in-title">Sign In</h1>
                    <p className="sign-in-subtitle">Please Enter Your Credentials</p>

                    {error && <div className="error-message" style={{ color: '#ff3b3b', marginBottom: '1rem', textAlign: 'center' }}>{error}</div>}

                    <form onSubmit={handleLogin} className="signin-form">
                        <div className="form-group">
                            <label className="input-label">Username</label>
                            <input
                                type="text"
                                className="styled-input"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
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
                                required
                            />
                        </div>

                        <div className="forgot-password-link">
                            <a href="#">Forget Password ?</a>
                        </div>

                        <button type="submit" className="login-btn" disabled={loading}>
                            {loading ? 'Logging in...' : 'Login'}
                        </button>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default Login;
