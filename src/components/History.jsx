import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Bell, User as UserIcon, Search, Filter, RotateCcw, MoreHorizontal, ChevronDown } from 'lucide-react';
import logo from '../assets/logo.png';
import './History.css';

const MOCK_CASES = [
    { id: '1001', name: 'Alice Smith', nic: '199012345678', status: 'Missing', date: '2023-10-12', time: '14:30' },
    { id: '1002', name: 'Bob Jones', nic: '198598765432', status: 'Investigating', date: '2023-10-15', time: '09:15' },
    { id: '1003', name: 'Charlie Brown', nic: '200134567890', status: 'Found', date: '2023-09-20', time: '11:45' },
    { id: '1004', name: 'Diana Prince', nic: '199511223344', status: 'Missing', date: '2023-10-18', time: '16:00' },
    { id: '1005', name: 'Evan Wright', nic: '198855667788', status: 'Found', date: '2023-08-10', time: '10:30' },
    { id: '1006', name: 'Fiona Gallagher', nic: '199999887766', status: 'Investigating', date: '2023-10-01', time: '08:00' },
];

const sortOptions = [
    { value: 'date-desc', label: 'Date (Newest)' },
    { value: 'date-asc', label: 'Date (Oldest)' },
    { value: 'name-asc', label: 'Name (A-Z)' },
    { value: 'name-desc', label: 'Name (Z-A)' },
];

const History = () => {
    const navigate = useNavigate();
    const [searchTerm, setSearchTerm] = useState('');
    const [sortBy, setSortBy] = useState('date-desc');
    const [isDropdownOpen, setIsDropdownOpen] = useState(false);
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

    const handleBack = () => {
        navigate(-1);
    };

    const getStatusClass = (status) => {
        switch(status.toLowerCase()) {
            case 'missing': return 'missing';
            case 'investigating': return 'investigating';
            case 'found': return 'found';
            default: return '';
        }
    };

    const filteredAndSortedCases = MOCK_CASES.filter(c => {
        const term = searchTerm.toLowerCase();
        return c.name.toLowerCase().includes(term) || 
               c.nic.includes(term) || 
               c.id.includes(term);
    }).sort((a, b) => {
        if (sortBy === 'date-desc') {
            return new Date(b.date + 'T' + b.time) - new Date(a.date + 'T' + a.time);
        } else if (sortBy === 'date-asc') {
            return new Date(a.date + 'T' + a.time) - new Date(b.date + 'T' + b.time);
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
            <header className="history-header">
                <div className="history-header-left">
                    <button className="back-btn" onClick={handleBack}>
                        <ArrowLeft size={24} />
                    </button>
                    <img src={logo} alt="Argus Logo" className="history-logo" />
                    <span className="history-title-text">ARGUS</span>
                </div>
                <div className="history-header-right">
                    <div className="user-profile">
                        <UserIcon size={24} fill="#00ff84" />
                        <span>John Doe</span>
                    </div>
                    <Bell size={24} className="notification-bell" fill="#ff3b3b" />
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
                    <div className="history-container-header">
                        <RotateCcw size={28} color="#00ff84" />
                        <h2>History</h2>
                    </div>
                    
                    <div className="cases-list">
                        {filteredAndSortedCases.map((c) => (
                            <div className="case-card" key={c.id}>
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
                                    <div className="status-badge">
                                        <div className={`status-dot ${getStatusClass(c.status)}`}></div>
                                        {c.status}
                                    </div>
                                    <button className="more-btn">
                                        <MoreHorizontal size={24} />
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </main>
        </div>
    );
};

export default History;
