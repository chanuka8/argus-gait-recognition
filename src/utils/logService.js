// Centralized logging service for ARGUS system
// All components import and call addLog() to record mandatory events.
// LogViewer listens for 'argus-log-update' custom events to refresh in real-time.

const STORAGE_KEY = 'argus_system_logs_live';

const getTimestamp = () => {
    return new Date().toISOString().slice(0, 19).replace('T', ' ');
};

export const getLogs = () => {
    try {
        const stored = localStorage.getItem(STORAGE_KEY);
        return stored ? JSON.parse(stored) : [];
    } catch {
        return [];
    }
};

export const addLog = (level, message, details = '', user = 'system', ip = 'localhost') => {
    const logs = getLogs();
    const newLog = {
        id: Date.now(),
        timestamp: getTimestamp(),
        level,        // 'info' | 'warning' | 'critical'
        category: 'SYSTEM',
        message,
        user,
        ip,
        details
    };

    const updated = [newLog, ...logs];
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));

    // Dispatch custom event so LogViewer can react in real-time
    window.dispatchEvent(new CustomEvent('argus-log-update', { detail: newLog }));

    return newLog;
};

export const clearLogs = () => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([]));
    window.dispatchEvent(new CustomEvent('argus-log-update'));
};

export const exportLogsCSV = () => {
    const logs = getLogs();
    const headers = 'ID,Timestamp,Level,Message,User,IP Address\n';
    const csvContent = logs.map(log =>
        `"${log.id}","${log.timestamp}","${log.level}","${log.message.replace(/"/g, '""')}","${log.user}","${log.ip}"`
    ).join('\n');

    const blob = new Blob([headers + csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `argus_audit_logs_${Date.now()}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
};
