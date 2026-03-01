import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { Shield, GraduationCap, Lock, Mail, ArrowRight, Info, AlertTriangle, User, Database } from 'lucide-react';
import { motion,AnimatePresence } from 'framer-motion';
import { cn } from '../lib/utils';
// import { theme } from '../styles/theme';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '../components/ui/card';
import { Input } from '../components/ui/input';
import { Tabs, TabsList, TabsTrigger } from '../components/ui/tabs';

const AuthPage = () => {
  // const { login, register } = useAuth();
  const { login } = useAuth();
  const [role, setRole] = useState('student');
  const [isRegister, setIsRegister] = useState(false);
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (isRegister && role === 'student' && password !== confirmPassword) {
      setError('Passwords do not match.');
      return;
    }

    setLoading(true);

    // let result;
    // if (isRegister && role === 'student') {
    //   result = register(fullName, email, password);
    // } else {
    //   result = login(email, password, role);
    // }

    // if (!result.success) {
    //   setError(result.message);
    //   setLoading(false);
    // }
    try {
      if (isRegister && role === 'student') {
        const response = await fetch("http://127.0.0.1:8080/api/register/", {
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          },
          body: JSON.stringify({
            username: fullName,   // sending fullName as username
            email: email,
            password: password
          })
        });
  
        const data = await response.json();
  
        if (!response.ok) {
          setError(data.message || "Registration failed");
          setLoading(false);
          return;
        }
  
        // success
        console.log("Registered:", data);
        setLoading(false);
  
      } else {
        // keep your login logic here
        const result = await login(email, password, role);
  
        if (!result.success) {
          setError(result.message);
          setLoading(false);
        }
      }
  
    } catch (err) {
      setError(err.message || "Server error. Please try again.");
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col relative overflow-hidden bg-[#F8FAFC]">
      {/* Background Dots Pattern */}
      <div className="absolute inset-0 z-0 pointer-events-none opacity-[0.35]">
        <div
          className="absolute inset-0"
          style={{
            backgroundImage: `radial-gradient(#94a3b8 0.5px, transparent 0.5px)`,
            backgroundSize: '24px 24px'
          }}
        />
      </div>

      {/* Header */}
      <header className="relative z-20 w-full px-12 py-8 flex justify-between items-center max-w-[1600px] mx-auto bg-white/50 backdrop-blur-sm border-b border-slate-100">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-[#1e3a8a] rounded-lg shadow-lg shadow-blue-900/20">
            <GraduationCap className="w-6 h-6 text-white" />
          </div>
          <span className="text-xl font-black text-slate-800 tracking-tight">University Research Data Management</span>
        </div>
        <button className="flex items-center gap-2 text-sm font-bold text-slate-600 hover:text-[#1e3a8a] transition-all">
          <div className="w-6 h-6 rounded-full bg-slate-200/50 flex items-center justify-center text-xs text-slate-600 border border-slate-300 font-black">?</div>
          Help & Support
        </button>
      </header>

      <main className="flex-1 flex items-center justify-center relative z-10 px-8 py-16">
        <div className="w-full max-w-7xl grid grid-cols-1 lg:grid-cols-2 gap-24 items-center">
          <div className="hidden lg:flex flex-col gap-12">
            <div className="space-y-8">
              <div
                className="inline-flex items-center gap-3 px-4 py-1.5 rounded-full text-[11px] font-black uppercase tracking-widest bg-blue-50/50 border border-blue-100 shadow-sm"
                style={{ color: '#1e3a8a' }}
              >
                <div className="relative flex h-2.5 w-2.5">
                  <div className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#1e3a8a] opacity-75"></div>
                  <div className="relative inline-flex rounded-full h-2.5 w-2.5 bg-[#1e3a8a]"></div>
                </div>
                System Operational
              </div>
              <h1 className="text-7xl font-black tracking-tight leading-[1.05] text-slate-900 drop-shadow-sm">
                Advance Your <br />
                Research <br />
                <span className="text-[#1e3a8a]">Securely & Efficiently</span>
              </h1>
              <p className="text-xl text-slate-500 max-w-lg leading-relaxed font-medium">
                Access the centralized database for academic publications, datasets, and collaborative projects. Secure login required.
              </p>
            </div>

            <div className="grid grid-cols-2 gap-8 max-w-xl">
              <Card className="p-8 rounded-[32px] border-slate-100 shadow-xl shadow-slate-200/50 bg-white group hover:scale-[1.02] transition-transform">
                <div className="w-12 h-12 rounded-2xl bg-slate-50 flex items-center justify-center mb-6 shadow-inner group-hover:bg-[#1e3a8a] transition-colors">
                  <Shield className="w-6 h-6 text-[#1e3a8a] group-hover:text-white transition-colors" />
                </div>
                <div>
                  <h3 className="font-black text-slate-900 text-base">Secure Access</h3>
                  <p className="text-sm font-medium text-slate-500 mt-2 leading-relaxed text-balance">End-to-end encrypted session management.</p>
                </div>
              </Card>
              <Card className="p-8 rounded-[32px] border-slate-100 shadow-xl shadow-slate-200/50 bg-white group hover:scale-[1.02] transition-transform">
                <div className="w-12 h-12 rounded-2xl bg-slate-50 flex items-center justify-center mb-6 shadow-inner group-hover:bg-[#1e3a8a] transition-colors">
                  <Database className="w-6 h-6 text-[#1e3a8a] group-hover:text-white transition-colors" />
                </div>
                <div>
                  <h3 className="font-black text-slate-900 text-base">Global Data</h3>
                  <p className="text-sm font-medium text-slate-500 mt-2 leading-relaxed text-balance">Connect with 50+ partner university archives.</p>
                </div>
              </Card>
            </div>
          </div>

          <motion.div
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            className="w-full max-w-md mx-auto"
          >
            <Card className="rounded-[40px] shadow-[0_45px_100px_-20px_rgba(30,58,138,0.12)] border-slate-200/60 overflow-hidden bg-white">
              <CardHeader className="text-center pt-14 pb-8 space-y-3">
                <CardTitle className="text-3xl font-black text-slate-900 tracking-tight">Secure Access Portal</CardTitle>
                <CardDescription className="text-slate-400 font-bold px-6">
                  {isRegister ? 'Enter your details to create a student account.' : 'Please sign in to access the research database.'}
                </CardDescription>
              </CardHeader>

              <CardContent className="px-10 pb-10 space-y-8">
                <div className="bg-slate-100/80 p-1.5 rounded-2xl">
                  <div className="grid grid-cols-2 gap-1">
                    <button
                      onClick={() => { setRole('student'); setIsRegister(false); setError(''); }}
                      className={cn(
                        "py-3 rounded-[14px] text-sm font-black transition-all",
                        role === 'student' ? "bg-white text-[#1e3a8a] shadow-md shadow-blue-900/5" : "text-slate-500 hover:text-slate-700 hover:bg-white/40"
                      )}
                    >
                      Student Access
                    </button>
                    <button
                      onClick={() => { setRole('admin'); setIsRegister(false); setError(''); }}
                      className={cn(
                        "py-3 rounded-[14px] text-sm font-black transition-all",
                        role === 'admin' ? "bg-white text-[#1e3a8a] shadow-md shadow-blue-900/5" : "text-slate-500 hover:text-slate-700 hover:bg-white/40"
                      )}
                    >
                      Admin Portal
                    </button>
                  </div>
                </div>

                <AnimatePresence mode="wait">
                  {error && (
                    <motion.div
                      initial={{ opacity: 0, y: -10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, y: -10 }}
                      className="p-4 bg-red-50 border border-red-100 rounded-2xl flex items-center gap-3 text-red-600 text-[13px] font-black"
                    >
                      <AlertTriangle className="w-5 h-5 shrink-0" />
                      {error}
                    </motion.div>
                  )}
                </AnimatePresence>

                <form onSubmit={handleSubmit} className="space-y-6">
                  <div className="space-y-6">
                    {isRegister && role === 'student' && (
                      <div className="space-y-2">
                        <label className="text-[11px] font-black text-slate-500 uppercase tracking-widest ml-1">Full Name</label>
                        <div className="relative">
                          <User className="absolute left-5 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                          <Input
                            type="text"
                            required
                            value={fullName}
                            onChange={(e) => setFullName(e.target.value)}
                            placeholder="John Doe"
                            className="pl-14 h-16 rounded-[18px] bg-slate-50/50 border-slate-200 focus:bg-white focus:ring-4 focus:ring-blue-100 transition-all font-bold text-slate-800"
                          />
                        </div>
                      </div>
                    )}

                    <div className="space-y-2">
                      <label className="text-[11px] font-black text-slate-500 uppercase tracking-widest ml-1">
                        {role === 'admin' ? 'Administrator Email' : 'University ID or Email'}
                      </label>
                      <div className="relative">
                        <Mail className="absolute left-5 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                        <Input
                          type="email"
                          required
                          value={email}
                          onChange={(e) => setEmail(e.target.value)}
                          placeholder={role === 'admin' ? "admin@university.edu" : "Enter your ID (e.g. 2024001)"}
                          className="pl-14 h-16 rounded-[18px] bg-slate-50/50 border-slate-200 focus:bg-white focus:ring-4 focus:ring-blue-100 transition-all font-bold text-slate-800"
                        />
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="flex items-center justify-between ml-1">
                        <label className="text-[11px] font-black text-slate-500 uppercase tracking-widest">Password</label>
                        {!isRegister && <button type="button" className="text-xs font-black text-[#1e3a8a] hover:underline opacity-90 transition-opacity">Forgot password?</button>}
                      </div>
                      <div className="relative">
                        <Lock className="absolute left-5 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                        <Input
                          type="password"
                          required
                          value={password}
                          onChange={(e) => setPassword(e.target.value)}
                          placeholder="••••••••"
                          className="pl-14 h-16 rounded-[18px] bg-slate-50/50 border-slate-200 focus:bg-white focus:ring-4 focus:ring-blue-100 transition-all font-bold text-slate-800"
                        />
                      </div>
                    </div>

                    {isRegister && role === 'student' && (
                      <div className="space-y-2">
                        <label className="text-[11px] font-black text-slate-500 uppercase tracking-widest ml-1">Confirm Password</label>
                        <div className="relative">
                          <Lock className="absolute left-5 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                          <Input
                            type="password"
                            required
                            value={confirmPassword}
                            onChange={(e) => setConfirmPassword(e.target.value)}
                            placeholder="••••••••"
                            className="pl-14 h-16 rounded-[18px] bg-slate-50/50 border-slate-200 focus:bg-white focus:ring-4 focus:ring-blue-100 transition-all font-bold text-slate-800"
                          />
                        </div>
                      </div>
                    )}
                  </div>

                  {!isRegister && (
                    <div className="flex items-center gap-3 ml-1">
                      <input type="checkbox" className="w-5 h-5 rounded-lg border-slate-300 text-[#1e3a8a] focus:ring-[#1e3a8a] cursor-pointer" id="remember" />
                      <label htmlFor="remember" className="text-sm font-bold text-slate-600 cursor-pointer select-none">Keep me signed in for 30 days</label>
                    </div>
                  )}

                  <Button
                    type="submit"
                    disabled={loading}
                    className="w-full h-16 rounded-2xl bg-[#1e3a8a] hover:bg-[#1a337a] text-white font-black text-lg shadow-2xl shadow-blue-950/20 transition-all active:scale-[0.98] disabled:opacity-50"
                  >
                    {loading ? (
                      <div className="flex items-center gap-3">
                        <motion.div animate={{ rotate: 360 }} transition={{ repeat: Infinity, duration: 1 }}>
                          <Shield className="w-5 h-5" />
                        </motion.div>
                        Processing...
                      </div>
                    ) : (
                      <span className="flex items-center gap-2">
                        {isRegister ? 'Log My Account' : 'Sign In To Portal'}
                        <ArrowRight className="w-5 h-5" />
                      </span>
                    )}
                  </Button>

                  <div className="flex items-start gap-4 p-5 bg-blue-50/50 rounded-2xl border border-blue-100/50">
                    <Info className="w-5 h-5 shrink-0 mt-0.5 text-[#1e3a8a]" />
                    <div className="text-xs text-slate-600 leading-relaxed font-medium">
                      <span className="font-black block mb-1 text-[#1e3a8a] uppercase tracking-widest">Session Security</span>
                      Your connection is end-to-end encrypted. Audit logs are recorded for institutional security compliance.
                    </div>
                  </div>
                </form>
              </CardContent>

              <CardFooter className="bg-slate-50/80 border-t border-slate-100 justify-center py-8">
                {role === 'student' ? (
                  <p className="text-sm font-bold text-slate-500">
                    {isRegister ? 'Already registered?' : 'Need institutional access?'}
                    <button
                      type="button"
                      onClick={() => { setIsRegister(!isRegister); setError(''); setConfirmPassword(''); }}
                      className="ml-2 text-[#1e3a8a] font-black hover:underline transition-all"
                    >
                      {isRegister ? 'Sign In' : 'Apply for access'}
                    </button>
                  </p>
                ) : (
                  <p className="text-[11px] font-black text-slate-400 uppercase tracking-[0.2em]">
                    Institutional Admin Portal
                  </p>
                )}
              </CardFooter>
            </Card>
          </motion.div>
        </div>
      </main>

      <footer className="relative z-20 w-full py-10 text-center text-slate-500 text-xs font-black border-t border-slate-100 bg-white tracking-widest uppercase">
        Copyright © 2024 University Research Board. All rights reserved. • Privacy Policy • Terms of Use
      </footer>
    </div>
  );
};

export default AuthPage;

