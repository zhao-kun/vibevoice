"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import { DialogSession, DialogLine } from "@/types/dialog";
import { useProject } from "@/lib/ProjectContext";
import { api } from "@/lib/api";
import type { DialogSession as ApiDialogSession } from "@/lib/api";

interface SessionContextType {
  sessions: DialogSession[];
  currentSession: DialogSession | null;
  selectSession: (sessionId: string) => Promise<void>;
  createSession: (name: string, description: string) => Promise<void>;
  deleteSession: (sessionId: string) => Promise<void>;
  updateSession: (sessionId: string, updates: Partial<DialogSession>) => Promise<void>;
  updateSessionDialogs: (sessionId: string, dialogLines: DialogLine[]) => Promise<void>;
  loading: boolean;
  error: string | null;
}

const SessionContext = createContext<SessionContextType | undefined>(undefined);

export function SessionProvider({ children }: { children: React.ReactNode }) {
  const { currentProject } = useProject();
  const [sessions, setSessions] = useState<DialogSession[]>([]);
  const [currentSession, setCurrentSession] = useState<DialogSession | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Helper: Convert backend dialog text to DialogLine array
  const parseDialogText = (dialogText: string): DialogLine[] => {
    const lines: DialogLine[] = [];
    const textLines = dialogText.trim().split('\n\n');

    textLines.forEach((line, index) => {
      const match = line.match(/^(Speaker \d+):\s*(.*)$/);
      if (match) {
        lines.push({
          id: `line-${Date.now()}-${index}`,
          speakerId: match[1],
          content: match[2],
        });
      }
    });

    return lines;
  };

  // Helper: Convert DialogLine array to backend dialog text
  const formatDialogText = (dialogLines: DialogLine[]): string => {
    return dialogLines
      .map(line => `${line.speakerId}: ${line.content}`)
      .join('\n\n');
  };

  // Helper: Convert backend session to frontend session
  const backendToFrontend = async (apiSession: ApiDialogSession, projectId: string): Promise<DialogSession> => {
    // Fetch dialog text if session has a text file
    let dialogLines: DialogLine[] = [];
    if (apiSession.text_filename) {
      try {
        const textResponse = await api.getSessionText(projectId, apiSession.session_id);
        dialogLines = parseDialogText(textResponse.dialog_text);
      } catch (err) {
        console.error('Failed to load dialog text:', err);
      }
    }

    return {
      id: apiSession.session_id,
      sessionId: apiSession.session_id,
      name: apiSession.name,
      description: apiSession.description,
      textFilename: apiSession.text_filename,
      dialogLines,
      createdAt: new Date(apiSession.created_at),
      updatedAt: new Date(apiSession.updated_at),
    };
  };

  // Load sessions from backend when project changes
  useEffect(() => {
    const loadSessions = async () => {
      if (!currentProject) {
        setSessions([]);
        setCurrentSession(null);
        return;
      }

      setLoading(true);
      setError(null);

      try {
        const response = await api.listSessions(currentProject.id);

        // Convert all sessions from backend format
        const frontendSessions = await Promise.all(
          response.sessions.map(s => backendToFrontend(s, currentProject.id))
        );

        setSessions(frontendSessions);

        // Restore current session from localStorage or use first session
        const savedCurrentSessionId = localStorage.getItem(`vibevoice_current_session_${currentProject.id}`);
        if (savedCurrentSessionId) {
          const current = frontendSessions.find((s: DialogSession) => s.id === savedCurrentSessionId);
          setCurrentSession(current || frontendSessions[0] || null);
        } else {
          setCurrentSession(frontendSessions[0] || null);
        }
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "Failed to load sessions";
        setError(errorMessage);
        console.error("Error loading sessions:", err);
      } finally {
        setLoading(false);
      }
    };

    loadSessions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentProject]);

  // Save current session ID to localStorage
  useEffect(() => {
    if (currentProject && currentSession) {
      localStorage.setItem(`vibevoice_current_session_${currentProject.id}`, currentSession.id);
    }
  }, [currentSession, currentProject]);

  const selectSession = async (sessionId: string): Promise<void> => {
    setError(null);
    const session = sessions.find((s) => s.id === sessionId);
    if (session) {
      setCurrentSession(session);
    }
  };

  const createSession = async (name: string, description: string): Promise<void> => {
    if (!currentProject) {
      throw new Error("No project selected");
    }

    setLoading(true);
    setError(null);

    try {
      // Create session on backend with empty dialog text
      const newApiSession = await api.createSession(currentProject.id, {
        name,
        description,
        dialog_text: "", // Empty initially
      });

      // Convert to frontend format
      const newSession = await backendToFrontend(newApiSession, currentProject.id);

      setSessions([...sessions, newSession]);
      setCurrentSession(newSession);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to create session";
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const deleteSession = async (sessionId: string): Promise<void> => {
    if (!currentProject) {
      throw new Error("No project selected");
    }

    if (sessions.length <= 1) {
      throw new Error("Cannot delete the last session");
    }

    setLoading(true);
    setError(null);

    try {
      await api.deleteSession(currentProject.id, sessionId);

      const newSessions = sessions.filter((s) => s.id !== sessionId);
      setSessions(newSessions);

      if (currentSession?.id === sessionId) {
        setCurrentSession(newSessions[0] || null);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to delete session";
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const updateSession = async (sessionId: string, updates: Partial<DialogSession>): Promise<void> => {
    if (!currentProject) {
      throw new Error("No project selected");
    }

    setError(null);

    // Update local state immediately for better UX
    const updatedSessions = sessions.map((s) =>
      s.id === sessionId
        ? { ...s, ...updates, updatedAt: new Date() }
        : s
    );
    setSessions(updatedSessions);

    if (currentSession?.id === sessionId) {
      setCurrentSession({ ...currentSession, ...updates, updatedAt: new Date() });
    }

    // Only sync name/description to backend (not dialogLines)
    if (updates.name !== undefined || updates.description !== undefined) {
      try {
        const updateData: { name?: string; description?: string } = {};
        if (updates.name !== undefined) updateData.name = updates.name;
        if (updates.description !== undefined) updateData.description = updates.description;

        const updatedApiSession = await api.updateSession(currentProject.id, sessionId, updateData);

        // Update with backend response
        const updatedSession = await backendToFrontend(updatedApiSession, currentProject.id);
        setSessions(sessions.map(s => s.id === sessionId ? updatedSession : s));
        if (currentSession?.id === sessionId) {
          setCurrentSession(updatedSession);
        }
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "Failed to update session";
        setError(errorMessage);
        throw err;
      }
    }
  };

  const updateSessionDialogs = async (sessionId: string, dialogLines: DialogLine[]): Promise<void> => {
    if (!currentProject) {
      throw new Error("No project selected");
    }

    setLoading(true);
    setError(null);

    try {
      // Convert dialog lines to backend text format
      const dialogText = formatDialogText(dialogLines);

      // Update on backend
      const updatedApiSession = await api.updateSession(currentProject.id, sessionId, {
        dialog_text: dialogText,
      });

      // Update local state
      const updatedSession = await backendToFrontend(updatedApiSession, currentProject.id);
      setSessions(sessions.map(s => s.id === sessionId ? updatedSession : s));

      if (currentSession?.id === sessionId) {
        setCurrentSession(updatedSession);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to update dialog";
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
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
        loading,
        error,
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
