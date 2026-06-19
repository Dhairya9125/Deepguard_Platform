"use client";

import React, { useEffect, useState } from "react";
import Navbar from "../components/Navbar";
import { User, Key, Bell, Shield, LogOut, Copy, Trash2, Plus } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useAuth } from "../contexts/AuthContext";
import { getProfile, updateProfile, listAPIKeys, createAPIKey, revokeAPIKey, APIKeyResponse } from "../lib/api";

export default function SettingsPage() {
  const { logout } = useAuth();
  
  const [activeTab, setActiveTab] = useState("profile");
  
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [initials, setInitials] = useState("");
  const [savingProfile, setSavingProfile] = useState(false);
  const [profileMsg, setProfileMsg] = useState({ type: "", text: "" });

  const [keys, setKeys] = useState<APIKeyResponse[]>([]);
  const [loadingKeys, setLoadingKeys] = useState(true);
  const [newKeyName, setNewKeyName] = useState("");
  const [isCreatingKey, setIsCreatingKey] = useState(false);

  useEffect(() => {
    document.body.style.background = "#050816";
    document.body.style.color = "#FFFFFF";
    return () => {
      document.body.style.background = "";
      document.body.style.color = "";
    };
  }, []);

  useEffect(() => {
    async function fetchData() {
      try {
        const profile = await getProfile();
        setUsername(profile.username);
        setEmail(profile.email);
        setInitials(profile.username.substring(0, 2).toUpperCase());
      } catch (err) {
        console.error("Failed to load profile", err);
      }
    }
    fetchData();
  }, []);

  useEffect(() => {
    if (activeTab === "keys") {
      fetchKeys();
    }
  }, [activeTab]);

  async function fetchKeys() {
    try {
      setLoadingKeys(true);
      const data = await listAPIKeys();
      setKeys(data);
    } catch (err) {
      console.error("Failed to load API keys", err);
    } finally {
      setLoadingKeys(false);
    }
  }

  async function handleProfileSave(e: React.FormEvent) {
    e.preventDefault();
    setSavingProfile(true);
    setProfileMsg({ type: "", text: "" });
    try {
      await updateProfile({ username, email });
      setProfileMsg({ type: "success", text: "Profile updated successfully." });
      setInitials(username.substring(0, 2).toUpperCase());
    } catch (err: unknown) {
      setProfileMsg({ type: "error", text: (err as Error).message || "Update failed." });
    } finally {
      setSavingProfile(false);
    }
  }

  async function handleCreateKey(e: React.FormEvent) {
    e.preventDefault();
    if (!newKeyName.trim()) return;
    setIsCreatingKey(true);
    try {
      await createAPIKey(newKeyName);
      setNewKeyName("");
      await fetchKeys();
    } catch (err: unknown) {
      alert((err as Error).message || "Failed to create API key.");
    } finally {
      setIsCreatingKey(false);
    }
  }

  async function handleRevokeKey(id: string) {
    if (!confirm("Are you sure you want to revoke this API key? This action cannot be undone.")) return;
    try {
      await revokeAPIKey(id);
      setKeys(keys.filter((k) => k.id !== id));
    } catch (err: unknown) {
      alert((err as Error).message || "Failed to revoke API key.");
    }
  }

  function handleCopy(text: string) {
    navigator.clipboard.writeText(text);
    // Optional: show a small toast
  }

  return (
    <div className="min-h-screen bg-[#050816] text-white font-sans overflow-x-hidden">
      <Navbar />

      <main className="pt-28 pb-20 px-6 max-w-7xl mx-auto flex flex-col md:flex-row gap-8">
        
        {/* Sidebar */}
        <motion.div 
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="w-full md:w-64 space-y-2 shrink-0"
        >
          <h1 className="text-3xl font-extrabold tracking-tight mb-6 px-2">Settings</h1>
          <div className="space-y-1">
            <button 
              onClick={() => setActiveTab("profile")}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl font-medium transition-colors ${activeTab === "profile" ? "bg-white/10 text-white" : "text-white/60 hover:text-white hover:bg-white/5"}`}
            >
              <User className="w-5 h-5" /> Profile Details
            </button>
            <button 
              onClick={() => setActiveTab("keys")}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl font-medium transition-colors ${activeTab === "keys" ? "bg-white/10 text-white" : "text-white/60 hover:text-white hover:bg-white/5"}`}
            >
              <Key className="w-5 h-5" /> API Keys
            </button>
            <button className="w-full flex items-center gap-3 px-4 py-3 hover:bg-white/5 rounded-xl text-white/60 hover:text-white transition-colors">
              <Bell className="w-5 h-5" /> Notifications
            </button>
            <button className="w-full flex items-center gap-3 px-4 py-3 hover:bg-white/5 rounded-xl text-white/60 hover:text-white transition-colors">
              <Shield className="w-5 h-5" /> Security
            </button>
          </div>
          <div className="pt-8 px-2 border-t border-white/10 mt-8">
            <button onClick={logout} className="w-full flex items-center gap-3 px-4 py-3 text-red-400 hover:bg-red-500/10 rounded-xl transition-colors">
              <LogOut className="w-5 h-5" /> Sign Out
            </button>
          </div>
        </motion.div>

        {/* Content Area */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="flex-1 max-w-3xl"
        >
          {activeTab === "profile" && (
            <div className="bg-white/5 border border-white/10 rounded-2xl p-8 backdrop-blur-md mb-8">
              <h2 className="text-xl font-bold mb-6">Profile Details</h2>
              
              <div className="flex items-center gap-6 mb-8">
                <div className="w-20 h-20 rounded-full bg-gradient-to-tr from-blue-600 to-cyan-400 flex items-center justify-center text-2xl font-bold shadow-lg shadow-blue-500/20">
                  {initials || "U"}
                </div>
                <div>
                  <button className="px-4 py-2 bg-white/10 hover:bg-white/20 rounded-lg text-sm font-medium transition-colors mb-2">Change Avatar</button>
                  <p className="text-xs text-white/40">JPG, GIF or PNG. 1MB max.</p>
                </div>
              </div>

              <form onSubmit={handleProfileSave} className="space-y-5">
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-white/70">Username</label>
                  <input 
                    type="text" 
                    value={username} 
                    onChange={(e) => setUsername(e.target.value)}
                    required
                    className="w-full bg-black/40 border border-white/10 rounded-lg px-4 py-2.5 outline-none focus:border-blue-500 transition-colors" 
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-sm font-medium text-white/70">Email Address</label>
                  <input 
                    type="email" 
                    value={email} 
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    className="w-full bg-black/40 border border-white/10 rounded-lg px-4 py-2.5 outline-none focus:border-blue-500 transition-colors" 
                  />
                </div>
                
                {profileMsg.text && (
                  <div className={`text-sm ${profileMsg.type === "error" ? "text-red-400" : "text-green-400"}`}>
                    {profileMsg.text}
                  </div>
                )}
                
                <div className="pt-4">
                  <button 
                    type="submit" 
                    disabled={savingProfile}
                    className="px-6 py-2.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-semibold transition-colors shadow-lg shadow-blue-500/20"
                  >
                    {savingProfile ? "Saving..." : "Save Changes"}
                  </button>
                </div>
              </form>
            </div>
          )}

          {activeTab === "keys" && (
            <div className="bg-white/5 border border-white/10 rounded-2xl p-8 backdrop-blur-md">
              <div className="flex flex-col md:flex-row justify-between items-start md:items-center mb-6 gap-4">
                <div>
                  <h2 className="text-xl font-bold">API Keys</h2>
                  <p className="text-sm text-white/50 mt-1">Authenticate scripts and apps with the DeepGuard API.</p>
                </div>
              </div>
              
              <form onSubmit={handleCreateKey} className="flex gap-3 mb-8">
                <input 
                  type="text" 
                  value={newKeyName}
                  onChange={(e) => setNewKeyName(e.target.value)}
                  placeholder="New API key name..." 
                  className="flex-1 bg-black/40 border border-white/10 rounded-lg px-4 py-2 text-sm outline-none focus:border-blue-500 transition-colors"
                  maxLength={50}
                  required
                />
                <button 
                  type="submit"
                  disabled={isCreatingKey}
                  className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 rounded-lg text-sm font-medium transition-colors"
                >
                  <Plus className="w-4 h-4" /> Generate Key
                </button>
              </form>
              
              <div className="space-y-3">
                {loadingKeys ? (
                  <div className="text-center py-8 text-white/40 text-sm animate-pulse">Loading keys...</div>
                ) : keys.length === 0 ? (
                  <div className="text-center py-8 text-white/40 text-sm bg-black/20 rounded-xl border border-white/5">No API keys found. Generate one above.</div>
                ) : (
                  <AnimatePresence>
                    {keys.map((k) => (
                      <motion.div 
                        key={k.id}
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        className="p-4 bg-black/40 border border-white/10 rounded-xl flex flex-col md:flex-row justify-between md:items-center gap-4 overflow-hidden"
                      >
                        <div className="flex-1 min-w-0">
                          <div className="font-semibold mb-1 truncate text-sm">{k.name}</div>
                          <div className="flex items-center gap-2 mb-2">
                            <code className="bg-white/10 px-2 py-1 rounded text-xs font-mono text-white/80">{k.key.substring(0, 16)}************************</code>
                            <button onClick={() => handleCopy(k.key)} className="p-1 hover:bg-white/10 rounded text-white/40 hover:text-white transition-colors" title="Copy full key">
                              <Copy className="w-3.5 h-3.5" />
                            </button>
                          </div>
                          <div className="text-xs text-white/40">
                            Created: {new Date(k.created_at).toLocaleDateString()} &bull; Last used: {k.last_used_at ? new Date(k.last_used_at).toLocaleDateString() : "Never"}
                          </div>
                        </div>
                        <button onClick={() => handleRevokeKey(k.id)} className="flex items-center gap-1.5 px-3 py-1.5 bg-red-500/10 hover:bg-red-500/20 text-red-400 rounded-lg text-xs font-medium transition-colors shrink-0 self-start md:self-auto">
                          <Trash2 className="w-3.5 h-3.5" /> Revoke
                        </button>
                      </motion.div>
                    ))}
                  </AnimatePresence>
                )}
              </div>
            </div>
          )}
        </motion.div>

      </main>
    </div>
  );
}
