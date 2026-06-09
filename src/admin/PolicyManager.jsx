import React, { useState, useEffect } from 'react';
import { Shield, Save, CheckCircle, RotateCcw, AlertOctagon, HelpCircle } from 'lucide-react';
import AdminHeader from './AdminHeader';
import './PolicyManager.css';

const PolicyManager = () => {
    const defaultPolicies = {
        alertSeverity: 'medium',
        dataRetention: 180,
        biometricMatch: 85,
        sessionTimeout: 30,
        autoArchiveCold: true,
        emailNotifications: false,
        backupFrequency: 'weekly'
    };

    const [policies, setPolicies] = useState(() => {
        const localPolicies = localStorage.getItem('argus_system_policies');
        return localPolicies ? JSON.parse(localPolicies) : defaultPolicies;
    });

    const [isSaving, setIsSaving] = useState(false);
    const [showToast, setShowToast] = useState(false);

    const handlePolicyChange = (field, value) => {
        setPolicies(prev => ({
            ...prev,
            [field]: value
        }));
    };

    const handleSave = (e) => {
        e.preventDefault();
        setIsSaving(true);

        // Simulate secure API handshake
        setTimeout(() => {
            localStorage.setItem('argus_system_policies', JSON.stringify(policies));
            setIsSaving(false);
            setShowToast(true);

            // Hide toast after 3 seconds
            setTimeout(() => {
                setShowToast(false);
            }, 3000);
        }, 1200);
    };

    const handleReset = () => {
        if (window.confirm('Reset all security policies to system defaults?')) {
            setPolicies(defaultPolicies);
        }
    };

    return (
        <div className="policy-manager-page">
            <AdminHeader />

            <main className="policy-content">
                <div className="policy-header-row">
                    <div className="title-group">
                        <h1>Security Policies & Parameters</h1>
                        <p>Configure facial scanning thresholds, data purge boundaries, and session timeout restrictions.</p>
                    </div>
                </div>

                <div className="policy-settings-layout">
                    <form onSubmit={handleSave} className="policy-form">
                        <div className="policy-section">
                            <h2>
                                <Shield size={18} />
                                <span>Threat & Biometric Detection</span>
                            </h2>
                            <p className="sec-desc">Calibrate facial matching tolerances and threat intelligence filters.</p>
                            
                            <div className="policy-control-item">
                                <div className="control-label-desc">
                                    <label>Facial Recognition Match Confidence (%)</label>
                                    <span>Lower values speed up matching but increase false positives. Recommendation: 85%.</span>
                                </div>
                                <div className="slider-control-row">
                                    <input 
                                        type="range" 
                                        min="50" 
                                        max="99" 
                                        value={policies.biometricMatch} 
                                        onChange={(e) => handlePolicyChange('biometricMatch', parseInt(e.target.value))}
                                        className="styled-slider"
                                    />
                                    <span className="slider-value-bubble">{policies.biometricMatch}%</span>
                                </div>
                            </div>

                            <div className="policy-control-item">
                                <div className="control-label-desc">
                                    <label>Automatic Incident Alert Level</label>
                                    <span>Define the lowest incident level that triggers automatic investigator notifications.</span>
                                </div>
                                <select 
                                    value={policies.alertSeverity} 
                                    onChange={(e) => handlePolicyChange('alertSeverity', e.target.value)}
                                    className="styled-select"
                                >
                                    <option value="low">Low (Notify all events)</option>
                                    <option value="medium">Medium (Notify suspect matches & high-risk missing)</option>
                                    <option value="high">High (Notify confirmed matches only)</option>
                                    <option value="critical">Critical (Notify core system errors & emergency threats)</option>
                                </select>
                            </div>
                        </div>

                        <div className="policy-section">
                            <h2>
                                <Shield size={18} />
                                <span>Data Archival & Lifespan</span>
                            </h2>
                            <p className="sec-desc">Set limits on database size and how long data remains in hot storage.</p>

                            <div className="policy-control-item">
                                <div className="control-label-desc">
                                    <label>Active Case Retention Limit (Days)</label>
                                    <span>Length of time telemetry coordinates are preserved before archival. Default: 180 days.</span>
                                </div>
                                <input 
                                    type="number" 
                                    min="30" 
                                    max="730" 
                                    value={policies.dataRetention}
                                    onChange={(e) => handlePolicyChange('dataRetention', parseInt(e.target.value))}
                                    className="styled-number-input"
                                    required
                                />
                            </div>

                            <div className="policy-control-item toggle-row">
                                <div className="control-label-desc">
                                    <label>Archive Cold Cases Automatically</label>
                                    <span>Archive missing person cases after 365 consecutive days with zero coordinate feeds.</span>
                                </div>
                                <label className="switch-toggle">
                                    <input 
                                        type="checkbox" 
                                        checked={policies.autoArchiveCold}
                                        onChange={(e) => handlePolicyChange('autoArchiveCold', e.target.checked)}
                                    />
                                    <span className="toggle-slider"></span>
                                </label>
                            </div>
                        </div>

                        <div className="policy-section">
                            <h2>
                                <Shield size={18} />
                                <span>Session Security & Access</span>
                            </h2>
                            <p className="sec-desc">Configure token expirations, session bounds, and outbound alerts.</p>

                            <div className="policy-control-item">
                                <div className="control-label-desc">
                                    <label>Operator Session Inactivity Limit (Minutes)</label>
                                    <span>Forced logout interval when zero clicks or cursor movements are detected.</span>
                                </div>
                                <input 
                                    type="number" 
                                    min="5" 
                                    max="240" 
                                    value={policies.sessionTimeout}
                                    onChange={(e) => handlePolicyChange('sessionTimeout', parseInt(e.target.value))}
                                    className="styled-number-input"
                                    required
                                />
                            </div>

                            <div className="policy-control-item toggle-row">
                                <div className="control-label-desc">
                                    <label>Email Alert Notifications on Critical Match</label>
                                    <span>Dispatch instant alerts to administration when biometric algorithms flag critical subjects.</span>
                                </div>
                                <label className="switch-toggle">
                                    <input 
                                        type="checkbox" 
                                        checked={policies.emailNotifications}
                                        onChange={(e) => handlePolicyChange('emailNotifications', e.target.checked)}
                                    />
                                    <span className="toggle-slider"></span>
                                </label>
                            </div>

                            <div className="policy-control-item">
                                <div className="control-label-desc">
                                    <label>Database Encryption Backup Schedule</label>
                                    <span>Automated snapshot triggers for full platform recovery.</span>
                                </div>
                                <select 
                                    value={policies.backupFrequency} 
                                    onChange={(e) => handlePolicyChange('backupFrequency', e.target.value)}
                                    className="styled-select"
                                >
                                    <option value="daily">Daily Snapshots</option>
                                    <option value="weekly">Weekly Snapshots</option>
                                    <option value="monthly">Monthly Snapshots</option>
                                </select>
                            </div>
                        </div>

                        <div className="policy-form-actions">
                            <button type="button" className="reset-policies-btn" onClick={handleReset}>
                                <RotateCcw size={16} />
                                <span>Restore Defaults</span>
                            </button>
                            
                            <button type="submit" className="save-policies-btn" disabled={isSaving}>
                                {isSaving ? (
                                    <>
                                        <div className="save-spinner"></div>
                                        <span>Encrypting & Writing...</span>
                                    </>
                                ) : (
                                    <>
                                        <Save size={16} />
                                        <span>Save Operational Policies</span>
                                    </>
                                )}
                            </button>
                        </div>
                    </form>

                    <div className="policy-sidebar-hint">
                        <div className="alert-box">
                            <AlertOctagon size={24} color="var(--status-investigating)" />
                            <h3>Important Core Notice</h3>
                            <p>Operational parameter modifications impact active scanning threads globally. Policies are cached locally and synced to remote nodes every 60 seconds.</p>
                        </div>
                        <div className="help-box">
                            <HelpCircle size={20} color="var(--sky)" />
                            <h3>Policy Hierarchy</h3>
                            <p>Argus follows strict access control levels. Only operators authenticated with root Admin flags can access or write to policy schemas.</p>
                        </div>
                    </div>
                </div>
            </main>

            {/* Glowing Success Toast */}
            {showToast && (
                <div className="policy-success-toast">
                    <CheckCircle size={20} color="var(--status-found)" />
                    <span>Argus security policies successfully compiled and deployed!</span>
                </div>
            )}
        </div>
    );
};

export default PolicyManager;
