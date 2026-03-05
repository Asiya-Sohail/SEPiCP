import React from 'react';
import { Routes, Route, Navigate, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { LayoutDashboard, Database, ArrowRight, BarChart3 } from 'lucide-react';
import Sidebar from './components/Sidebar';
import TopBar from './components/TopBar';
import AuthPage from './pages/AuthPage';
import UploadPage from './pages/UploadPage';
import AnalysisPage from './pages/AnalysisPage';
import ResultsPage from './pages/ResultsPage';
import HistoryPage from './pages/HistoryPage';
import Dashboard from './pages/Dashboard';
import AdminPanel from './pages/AdminPanel';
import { useAuth } from './context/AuthContext';
import { theme } from './styles/theme';

const Layout = ({ children }) => {
  const location = useLocation();

  return (
    <div className="flex min-h-screen relative overflow-hidden bg-white"> {/* Changed background color */}
      <Sidebar />
      <div className="flex-1 flex flex-col h-screen overflow-hidden"> {/* New wrapper div */}
        <TopBar /> {/* Added TopBar */}
        <main className="flex-1 overflow-y-auto p-4 md:p-8 lg:p-10 relative bg-[#F8FAFC]">
          {/* Global Dot Grid Pattern - inside main to not cover sidebar/topbar */}
          <div className="absolute inset-0 z-0 pointer-events-none opacity-40">
            <div
              className="absolute inset-0"
              style={{
                backgroundImage: `radial-gradient(#94a3b8 0.5px, transparent 0.5px)`,
                backgroundSize: '24px 24px'
              }}
            />
          </div>

          <div className="max-w-[1400px] mx-auto relative z-10"> {/* Adjusted max-width */}
            <AnimatePresence mode="wait">
              <motion.div
                key={location.pathname}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.3 }}
              >
                {children}
              </motion.div>
            </AnimatePresence>
          </div>
        </main>
      </div>
    </div>
  );
};

export default function App() {
  const { isAuthenticated, user } = useAuth();

  if (!isAuthenticated) {
    return <AuthPage />;
  }

  return (
    <Routes>
      <Route path="/" element={<Layout><Dashboard /></Layout>} />
      <Route path="/upload" element={<Layout><UploadPage /></Layout>} />
      <Route path="/analysis" element={<Layout><AnalysisPage /></Layout>} />
      <Route path="/results" element={<Layout><ResultsPage /></Layout>} />
      <Route path="/history" element={<Layout><HistoryPage /></Layout>} />
      {user?.role === 'admin' && (
        <>
          <Route path="/admin" element={<Layout><AdminPanel /></Layout>} />
          <Route path="/logs" element={<Layout><AdminPanel /></Layout>} />
        </>
      )}
      <Route path="/settings" element={<Layout><Dashboard /></Layout>} />
      <Route path="*" element={<Navigate to="/" />} />
    </Routes>
  );
}


