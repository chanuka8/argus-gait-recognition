import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import Login from './components/Login';
import ProtectedRoute from './components/ProtectedRoute';
import History from './components/History';
import ReportCase from './components/ReportCase';
import CaseDetails from './components/CaseDetails';
import AdminDashboard from './admin/AdminDashboard';
import UserManagement from './admin/UserManagement';
import LogViewer from './admin/LogViewer';
import SurveillanceFeeds from './admin/SurveillanceFeeds';
import PolicyManager from './admin/PolicyManager';
import CasesManagement from './admin/CasesManagement';
import './App.css';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Login />} />
      <Route
        path="/dashboard"
        element={
          <ProtectedRoute allowedRole="investigator">
            <Dashboard />
          </ProtectedRoute>
        }
      />
      <Route
        path="/history"
        element={
          <ProtectedRoute allowedRole="investigator">
            <History />
          </ProtectedRoute>
        }
      />
      <Route
        path="/report-case"
        element={
          <ProtectedRoute allowedRole="investigator">
            <ReportCase />
          </ProtectedRoute>
        }
      />
      <Route
        path="/case/:id"
        element={
          <ProtectedRoute allowedRole="investigator">
            <CaseDetails />
          </ProtectedRoute>
        }
      />
      <Route
        path="/admin/dashboard"
        element={
          <ProtectedRoute allowedRole="admin">
            <AdminDashboard />
          </ProtectedRoute>
        }
      />
      <Route
        path="/admin/users"
        element={
          <ProtectedRoute allowedRole="admin">
            <UserManagement />
          </ProtectedRoute>
        }
      />
      <Route
        path="/admin/cases"
        element={
          <ProtectedRoute allowedRole="admin">
            <CasesManagement />
          </ProtectedRoute>
        }
      />
      <Route
        path="/admin/logs"
        element={
          <ProtectedRoute allowedRole="admin">
            <LogViewer />
          </ProtectedRoute>
        }
      />
      <Route
        path="/admin/surveillance"
        element={
          <ProtectedRoute allowedRole="admin">
            <SurveillanceFeeds />
          </ProtectedRoute>
        }
      />
      <Route
        path="/admin/policies"
        element={
          <ProtectedRoute allowedRole="admin">
            <PolicyManager />
          </ProtectedRoute>
        }
      />
    </Routes>
  );
}

export default App;
