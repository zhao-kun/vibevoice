"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import { DialogSession } from "@/types/dialog";
import { useProject } from "@/lib/ProjectContext";

interface SessionContextType {
  sessions: DialogSession[];
  currentSession: DialogSession | null;
  selectSession: (sessionId: string) => void;
  createSession: (name: string, description: string) => void;
  deleteSession: (sessionId: string) => void;
  updateSession: (sessionId: string, updates: Partial<DialogSession>) => void;
  updateSessionDialogs: (sessionId: string, dialogLines: any[]) => void;
}

const SessionContext = createContext<SessionContextType | undefined>(undefined);

export function SessionProvider({ children }: { children: React.ReactNode }) {
  const { currentProject } = useProject();
  const [sessions, setSessions] = useState<DialogSession[]>([]);
  const [currentSession, setCurrentSession] = useState<DialogSession | null>(null);

  // Load sessions from localStorage when project changes
  useEffect(() => {
    if (currentProject) {
      const savedSessions = localStorage.getItem(`vibevoice_sessions_${currentProject.id}`);
      const savedCurrentSessionId = localStorage.getItem(`vibevoice_current_session_${currentProject.id}`);

      if (savedSessions) {
        const parsed = JSON.parse(savedSessions);
        const sessionsWithDates = parsed.map((s: any) => ({
          ...s,
          createdAt: new Date(s.createdAt),
          updatedAt: new Date(s.updatedAt),
        }));
        setSessions(sessionsWithDates);

        if (savedCurrentSessionId) {
          const current = sessionsWithDates.find((s: DialogSession) => s.id === savedCurrentSessionId);
          if (current) {
            setCurrentSession(current);
          } else {
            setCurrentSession(sessionsWithDates[0] || null);
          }
        } else {
          setCurrentSession(sessionsWithDates[0] || null);
        }
      } else {
        // Create a default session on first load
        const defaultSession: DialogSession = {
          id: "session-1",
          name: "Dialog Session 1",
          description: "Default dialog session",
          dialogLines: [],
          createdAt: new Date(),
          updatedAt: new Date(),
        };
        setSessions([defaultSession]);
        setCurrentSession(defaultSession);
        localStorage.setItem(`vibevoice_sessions_${currentProject.id}`, JSON.stringify([defaultSession]));
        localStorage.setItem(`vibevoice_current_session_${currentProject.id}`, defaultSession.id);
      }
    } else {
      setSessions([]);
      setCurrentSession(null);
    }
  }, [currentProject]);

  // Save sessions to localStorage whenever they change
  useEffect(() => {
    if (currentProject && sessions.length > 0) {
      localStorage.setItem(`vibevoice_sessions_${currentProject.id}`, JSON.stringify(sessions));
    }
  }, [sessions, currentProject]);

  // Save current session to localStorage
  useEffect(() => {
    if (currentProject && currentSession) {
      localStorage.setItem(`vibevoice_current_session_${currentProject.id}`, currentSession.id);
    }
  }, [currentSession, currentProject]);

  const selectSession = (sessionId: string) => {
    const session = sessions.find((s) => s.id === sessionId);
    if (session) {
      setCurrentSession(session);
    }
  };

  const createSession = (name: string, description: string) => {
    const newSession: DialogSession = {
      id: `session-${Date.now()}`,
      name,
      description,
      dialogLines: [],
      createdAt: new Date(),
      updatedAt: new Date(),
    };
    setSessions([...sessions, newSession]);
    setCurrentSession(newSession);
  };

  const deleteSession = (sessionId: string) => {
    if (sessions.length <= 1) {
      alert("Cannot delete the last session");
      return;
    }

    const newSessions = sessions.filter((s) => s.id !== sessionId);
    setSessions(newSessions);

    if (currentSession?.id === sessionId) {
      setCurrentSession(newSessions[0]);
    }
  };

  const updateSession = (sessionId: string, updates: Partial<DialogSession>) => {
    const updatedSessions = sessions.map((s) =>
      s.id === sessionId
        ? { ...s, ...updates, updatedAt: new Date() }
        : s
    );
    setSessions(updatedSessions);

    if (currentSession?.id === sessionId) {
      setCurrentSession({ ...currentSession, ...updates, updatedAt: new Date() });
    }
  };

  const updateSessionDialogs = (sessionId: string, dialogLines: any[]) => {
    const updatedSessions = sessions.map((s) =>
      s.id === sessionId
        ? { ...s, dialogLines, updatedAt: new Date() }
        : s
    );
    setSessions(updatedSessions);

    if (currentSession?.id === sessionId) {
      setCurrentSession({ ...currentSession, dialogLines, updatedAt: new Date() });
    }
  };

  return (
    <SessionContext.Provider
      value={{
        sessions,
        currentSession,
        selectSession,
        createSession,
        deleteSession,
        updateSession,
        updateSessionDialogs,
      }}
    >
      {children}
    </SessionContext.Provider>
  );
}

export function useSession() {
  const context = useContext(SessionContext);
  if (context === undefined) {
    throw new Error("useSession must be used within a SessionProvider");
  }
  return context;
}
