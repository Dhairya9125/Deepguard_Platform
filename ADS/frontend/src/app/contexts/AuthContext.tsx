"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import { User, login as apiLogin, register as apiRegister, getMe } from "../lib/api";

interface AuthContextType {
  user: User | null;
  token: string | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (username: string, firstName: string, email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check local storage on mount
    const storedToken = localStorage.getItem("dg_token");
    if (storedToken) {
      Promise.resolve().then(() => setToken(storedToken));
      getMe()
        .then((userData) => {
          setUser(userData);
        })
        .catch(() => {
          // Token is invalid or expired
          localStorage.removeItem("dg_token");
          setToken(null);
          setUser(null);
        })
        .finally(() => {
          setLoading(false);
        });
    } else {
      Promise.resolve().then(() => setLoading(false));
    }
  }, []);

  const login = async (email: string, password: string) => {
    const data = await apiLogin(email, password);
    localStorage.setItem("dg_token", data.access_token);
    setToken(data.access_token);
    setUser(data.user);
  };

  const register = async (username: string, firstName: string, email: string, password: string) => {
    await apiRegister(username, firstName, email, password);
    // Auto login after register
    await login(email, password);
  };

  const logout = () => {
    localStorage.removeItem("dg_token");
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, token, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
