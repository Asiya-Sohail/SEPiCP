import React, { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext(undefined);

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [registeredUsers, setRegisteredUsers] = useState([]);
  const [activities, setActivities] = useState([]);

  useEffect(() => {
    const savedUser = localStorage.getItem('research_user');
    if (savedUser) {
      setUser(JSON.parse(savedUser));
    }

    const savedUsers = localStorage.getItem('registered_users');
    if (savedUsers) {
      setRegisteredUsers(JSON.parse(savedUsers));
    }

    const savedActivities = localStorage.getItem('research_activities');
    if (savedActivities) {
      setActivities(JSON.parse(savedActivities));
    }
  }, []);

  const logActivity = (type, details, targetUser = null) => {
    const activeUser = targetUser || user;
    if (!activeUser) return;

    const newActivity = {
      id: `act-${Date.now()}`,
      userId: activeUser.id,
      userName: activeUser.name,
      userRole: activeUser.role,
      type, // 'login', 'logout', 'analysis', 'upload'
      details,
      timestamp: new Date().toISOString(),
    };

    const updatedActivities = [newActivity, ...activities];
    setActivities(updatedActivities);
    localStorage.setItem('research_activities', JSON.stringify(updatedActivities));
  };

  const register = async (name, email, password) => {
    try {
      const response = await fetch("http://127.0.0.1:8000/api/register/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          username: email.split('@')[0], // Use email prefix as username for now
          email: email,
          password: password,
          first_name: name
        })
      });

      const data = await response.json();

      if (!response.ok) {
        console.error("Registration failed with status", response.status, "and data:", data);

        // Check for specific field errors from Django Rest Framework
        const errorMsg =
          data.password?.[0] ||
          data.email?.[0] ||
          data.username?.[0] ||
          data.non_field_errors?.[0] ||
          data.message ||
          "Registration failed: Please check your input fields.";

        return { success: false, message: errorMsg };
      }

      // Automatically log in after registration might need another call if backend doesn't return tokens
      return login(email, password, 'student');
    } catch (err) {
      return { success: false, message: "Server error. Please try again." };
    }
  };

  const login = async (email, password, role) => {
    try {
      const response = await fetch("http://127.0.0.1:8000/api/login/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password })
      });

      const data = await response.json();

      if (!response.ok) {
        return { success: false, message: "Invalid credentials." };
      }

      const userData = {
        name: email.split('@')[0], // Fallback if name not in JWT
        email: email,
        role: role, // You might want to get this from the backend/JWT
      };

      setUser(userData);
      localStorage.setItem('research_user', JSON.stringify(userData));
      localStorage.setItem('research_token', data.access);
      localStorage.setItem('research_refresh_token', data.refresh);

      logActivity('login', `${role} logged into dashboard`, userData);
      return { success: true };
    } catch (err) {
      return { success: false, message: "Server error. Please try again." };
    }
  };

  const logout = () => {
    logActivity('logout', 'User logged out');
    setUser(null);
    localStorage.removeItem('research_user');
    localStorage.removeItem('research_token');
  };

  return (
    <AuthContext.Provider value={{
      user,
      login,
      register,
      logout,
      logActivity,
      activities,
      allUsers: registeredUsers,
      isAuthenticated: !!user
    }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

