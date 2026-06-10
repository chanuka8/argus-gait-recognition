import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Bell, User as UserIcon, Search, Filter, RotateCcw, MoreHorizontal, ChevronDown, XCircle } from 'lucide-react';
import logo from '../assets/logo.png';
import Notifications from './Notifications';
import UserProfileModal from './UserProfileModal';
import { db } from '../firebaseConfig';
import { collection, getDocs } from 'firebase/firestore';
import './History.css';

const sortOptions = [
    { value: 'date-desc', label: 'Date (Newest)' },
    { value: 'date-asc', label: 'Date (Oldest)' },
    { value: 'name-asc', label: 'Name (A-Z)' },
    { value: 'name-desc', label: 'Name (Z-A)' },
];

const History = () => {
    const navigate = useNavigate();
    const [showNotifications, setShowNotifications] = useState(false);
    const [showProfile, setShowProfile] = useState(false);
    const [searchTerm, setSearchTerm] = useState('');
    const [sortBy, setSortBy] = useState('date-desc');
    const [isDropdownOpen, setIsDropdownOpen] = useState(false);
    const [cases, setCases] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState(null);
    const dropdownRef = useRef(null);

    useEffect(() => {
        const handleClickOutside = (event) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
                setIsDropdownOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    useEffect(() => {
        const fetchCases = async () => {
            try {
                setIsLoading(true);
                const querySnapshot = await getDocs(collection(db, 'victims'));
                const casesData = querySnapshot.docs.map(doc => {
                    const data = doc.data();
                    let createdDate = new Date();
                    if (data.createdAt) {
                        if (typeof data.createdAt.toDate === 'function') {
                            createdDate = data.createdAt.toDate();
                        } else if (data.createdAt.seconds) {
                            createdDate = new Date(data.createdAt.seconds * 1000);
                        } else {
                            createdDate = new Date(data.createdAt);
                        }
                    }
                    return {
                        id: data.caseId || doc.id,
                        name: data.name || 'Unnamed Case',
                        nic: data.nic || '',
                        status: data.status || 'Investigating',
                        createdDate: createdDate,
                    };
                });
                setCases(casesData);
            } catch (err) {
                console.error("Error fetching cases:", err);
                setError("Failed to load history cases.");
            } finally {
                setIsLoading(false);
            }
        };

        fetchCases();
    }, []);

    const handleBack = () => {
        navigate(-1);
    };

    const handleClose = () => {
        navigate('/dashboard');
    };

    const getStatusClass = (status) => {
        if (!status) return '';
        switch(status.toLowerCase()) {
            case 'investigating': return 'investigating';
            case 'cold': return 'cold';
            case 'found': return 'found';
            case 'closed': return 'closed';
            default: return '';
        }
    };

    const filteredAndSortedCases = cases.filter(c => {
        const term = searchTerm.toLowerCase();
        return c.name.toLowerCase().includes(term) || 
               c.nic.toLowerCase().includes(term) || 
               c.id.toLowerCase().includes(term);
    }).sort((a, b) => {
        if (sortBy === 'date-desc') {
            return b.createdDate - a.createdDate;
        } else if (sortBy === 'date-asc') {
            return a.createdDate - b.createdDate;
        } else if (sortBy === 'name-asc') {
            return a.name.localeCompare(b.name);
        } else if (sortBy === 'name-desc') {
            return b.name.localeCompare(a.name);
        }
        return 0;
    });

    const currentSortLabel = sortOptions.find(opt => opt.value === sortBy)?.label || 'Sort by';

    return (
        <div className="history-page">
            <Notifications isOpen={showNotifications} onClose={() => setShowNotifications(false)} />
            <UserProfileModal isOpen={showProfile} onClose={() => setShowProfile(false)} />
            
            <header className="history-header">
                <div className="history-header-left">
                    <button className="history-back-btn" onClick={handleBack}>
                        <ArrowLeft size={24} />
                    </button>
                    <img src={logo} alt="Argus Logo" className="history-logo" />
                    <span className="history-title-text">ARGUS</span>
                </div>
                <div className="history-header-right">
                    <div className="user-profile" onClick={() => setShowProfile(true)} style={{ cursor: 'pointer' }}>
                        <UserIcon size={22} fill="#d6e4ea" color="#d6e4ea" />
                        <span>John Doe</span>
                    </div>
                    <Bell 
                        size={22} 
                        className="notification-bell" 
                        fill="#5ce1e6" 
                        color="#5ce1e6"
                        onClick={() => setShowNotifications(true)}
                        style={{ cursor: 'pointer' }}
                    />
                </div>
            </header>

            <div className="history-controls">
                <div className="search-bar">
                    <Search size={20} color="#6b7280" />
                    <input 
                        type="text" 
                        placeholder="Search by Name, NIC, Case ID" 
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                    />
                </div>
                <div className="custom-dropdown-container" ref={dropdownRef}>
                    <div 
                        className={`custom-dropdown-trigger ${isDropdownOpen ? 'active' : ''}`}
                        onClick={() => setIsDropdownOpen(!isDropdownOpen)}
                    >
                        <Filter size={20} />
                        <span>Sort by: {currentSortLabel}</span>
                        <ChevronDown size={20} className={`chevron-icon ${isDropdownOpen ? 'open' : ''}`} />
                    </div>
                    
                    {isDropdownOpen && (
                        <div className="custom-dropdown-menu">
                            {sortOptions.map(option => (
                                <div 
                                    key={option.value}
                                    className={`custom-dropdown-item ${sortBy === option.value ? 'selected' : ''}`}
                                    onClick={() => {
                                        setSortBy(option.value);
                                        setIsDropdownOpen(false);
                                    }}
                                >
                                    {option.label}
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>

            <main className="history-content">
                <div className="history-container">
                    <button className="history-close-btn" onClick={handleClose}>
                        <XCircle size={28} fill="#E53935" color="#ffffff" />
                    </button>
                    
                    <div className="history-container-header">
                        <RotateCcw size={24} color="#4ab8bd" />
                        <h2>History</h2>
                    </div>
                    
                    <div className="cases-list">
                        {isLoading ? (
                            <div className="history-loading-container">
                                <div className="history-spinner"></div>
                                <h3>Loading Case History...</h3>
                                <p>Securing details from connection feed...</p>
                            </div>
                        ) : error ? (
                            <div className="history-error-container">
                                <XCircle size={40} fill="#E53935" color="#ffffff" />
                                <h3>Error Fetching Cases</h3>
                                <p>{error}</p>
                            </div>
                        ) : filteredAndSortedCases.length === 0 ? (
                            <div className="history-empty-container">
                                <Search size={40} color="#5ce1e6" />
                                <h3>No Cases Found</h3>
                                <p>{searchTerm ? "No cases match your search query." : "There are currently no reported cases."}</p>
                            </div>
                        ) : (
                            filteredAndSortedCases.map((c) => (
                                <div 
                                    className="case-card" 
                                    key={c.id} 
                                    onClick={() => navigate(`/case/${c.id}`)}
                                    style={{ cursor: 'pointer' }}
                                >
                                    <div className="case-card-left">
                                        <div className="case-avatar">
                                            <UserIcon size={48} />
                                        </div>
                                        <div className="case-details">
                                            <span>Case id : {c.id}</span>
                                            <span>Case Name : {c.name}</span>
                                        </div>
                                    </div>
                                    <div className="case-card-right">
                                        <div className={`status-badge ${getStatusClass(c.status)}`}>
                                            <div className={`status-dot ${getStatusClass(c.status)}`}></div>
                                            <span className="status-text">{c.status}</span>
                                        </div>
                                        <button 
                                            className="more-btn"
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                navigate(`/case/${c.id}`);
                                            }}
                                        >
                                            <MoreHorizontal size={24} />
                                        </button>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </main>
        </div>
    );
};

export default History;
