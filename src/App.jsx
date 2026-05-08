import React from 'react';
import { Routes, Route } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import Login from './components/Login';
import ProtectedRoute from './components/ProtectedRoute';
import History from './components/History';
import ReportCase from './components/ReportCase';
import CaseDetails from './components/CaseDetails';
import './App.css';

function App() {
  return (
    <Routes>
      <Route path="/" element={<Login />} />
      <Route
        path="/dashboard"
        element={
          <ProtectedRoute>
            <Dashboard />
          </ProtectedRoute>
        }
      />
      <Route
        path="/history"
        element={
          <ProtectedRoute>
            <History />
          </ProtectedRoute>
        }
      />
      <Route
        path="/report-case"
        element={
          <ProtectedRoute>
            <ReportCase />
          </ProtectedRoute>
        }
      />
      <Route
        path="/case/:id"
        element={
          <ProtectedRoute>
            <CaseDetails />
          </ProtectedRoute>
        }
      />
    </Routes>
  );
}

export default App;
