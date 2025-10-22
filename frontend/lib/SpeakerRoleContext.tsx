"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import { SpeakerRole, VoiceFile } from "@/types/speaker";

interface SpeakerRoleContextType {
  speakerRoles: SpeakerRole[];
  addSpeakerRole: () => void;
  updateSpeakerRole: (id: string, updates: Partial<SpeakerRole>) => void;
  deleteSpeakerRole: (id: string) => void;
  uploadVoiceFile: (id: string, file: File) => Promise<void>;
  removeVoiceFile: (id: string) => void;
  saveSpeakerRoles: () => void;
  hasUnsavedChanges: boolean;
}

const SpeakerRoleContext = createContext<SpeakerRoleContextType | undefined>(undefined);

export function SpeakerRoleProvider({ children, projectId }: { children: React.ReactNode; projectId: string }) {
  const [speakerRoles, setSpeakerRoles] = useState<SpeakerRole[]>([]);
  const [savedSpeakerRoles, setSavedSpeakerRoles] = useState<SpeakerRole[]>([]);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);

  // Generate speaker ID based on index
  const generateSpeakerId = (index: number): string => {
    return `Speaker ${index + 1}`;
  };

  // Reindex all speaker IDs to maintain sequential order
  const reindexSpeakerRoles = (roles: SpeakerRole[]): SpeakerRole[] => {
    return roles.map((role, index) => ({
      ...role,
      speakerId: generateSpeakerId(index),
    }));
  };

  // Load speaker roles from localStorage on mount
  useEffect(() => {
    const storageKey = `vibevoice_speaker_roles_${projectId}`;
    const saved = localStorage.getItem(storageKey);

    if (saved) {
      const parsed = JSON.parse(saved);
      setSpeakerRoles(parsed);
      setSavedSpeakerRoles(JSON.parse(JSON.stringify(parsed))); // Deep copy
    } else {
      // Initialize with one empty speaker role
      const defaultRole: SpeakerRole = {
        id: `role-${Date.now()}`,
        speakerId: "Speaker 1",
        description: "",
        voiceFile: null,
      };
      setSpeakerRoles([defaultRole]);
      setSavedSpeakerRoles([defaultRole]);
    }
  }, [projectId]);

  // Check for unsaved changes
  useEffect(() => {
    const hasChanges = JSON.stringify(speakerRoles) !== JSON.stringify(savedSpeakerRoles);
    setHasUnsavedChanges(hasChanges);
  }, [speakerRoles, savedSpeakerRoles]);

  const addSpeakerRole = () => {
    const newRole: SpeakerRole = {
      id: `role-${Date.now()}`,
      speakerId: generateSpeakerId(speakerRoles.length),
      description: "",
      voiceFile: null,
    };
    setSpeakerRoles([...speakerRoles, newRole]);
  };

  const updateSpeakerRole = (id: string, updates: Partial<SpeakerRole>) => {
    setSpeakerRoles(roles =>
      roles.map(role =>
        role.id === id ? { ...role, ...updates } : role
      )
    );
  };

  const deleteSpeakerRole = (id: string) => {
    const updatedRoles = speakerRoles.filter(role => role.id !== id);
    // Reindex speaker IDs after deletion
    const reindexedRoles = reindexSpeakerRoles(updatedRoles);
    setSpeakerRoles(reindexedRoles);
  };

  const uploadVoiceFile = async (id: string, file: File): Promise<void> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();

      reader.onload = (e) => {
        const dataUrl = e.target?.result as string;
        const voiceFile: VoiceFile = {
          name: file.name,
          size: file.size,
          type: file.type,
          dataUrl,
          uploadedAt: new Date().toISOString(),
        };

        updateSpeakerRole(id, { voiceFile });
        resolve();
      };

      reader.onerror = () => {
        reject(new Error("Failed to read file"));
      };

      reader.readAsDataURL(file);
    });
  };

  const removeVoiceFile = (id: string) => {
    updateSpeakerRole(id, { voiceFile: null });
  };

  const saveSpeakerRoles = () => {
    const storageKey = `vibevoice_speaker_roles_${projectId}`;
    localStorage.setItem(storageKey, JSON.stringify(speakerRoles));
    setSavedSpeakerRoles(JSON.parse(JSON.stringify(speakerRoles))); // Deep copy
    setHasUnsavedChanges(false);
  };

  return (
    <SpeakerRoleContext.Provider
      value={{
        speakerRoles,
        addSpeakerRole,
        updateSpeakerRole,
        deleteSpeakerRole,
        uploadVoiceFile,
        removeVoiceFile,
        saveSpeakerRoles,
        hasUnsavedChanges,
      }}
    >
      {children}
    </SpeakerRoleContext.Provider>
  );
}

export function useSpeakerRole() {
  const context = useContext(SpeakerRoleContext);
  if (context === undefined) {
    throw new Error("useSpeakerRole must be used within a SpeakerRoleProvider");
  }
  return context;
}
