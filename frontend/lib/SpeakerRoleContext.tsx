"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import { SpeakerRole } from "@/types/speaker";
import { api } from "@/lib/api";
import type { Speaker } from "@/lib/api";

interface SpeakerRoleContextType {
  speakerRoles: SpeakerRole[];
  addSpeakerRole: () => Promise<void>;
  updateSpeakerRole: (speakerId: string, updates: Partial<SpeakerRole>) => Promise<void>;
  deleteSpeakerRole: (speakerId: string) => Promise<void>;
  uploadVoiceFile: (speakerId: string, file: File) => Promise<void>;
  hasUnsavedChanges: boolean;
  loading: boolean;
  error: string | null;
}

const SpeakerRoleContext = createContext<SpeakerRoleContextType | undefined>(undefined);

export function SpeakerRoleProvider({ children, projectId }: { children: React.ReactNode; projectId: string }) {
  const [speakerRoles, setSpeakerRoles] = useState<SpeakerRole[]>([]);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Convert backend Speaker to frontend SpeakerRole
  const backendToFrontend = (speaker: Speaker): SpeakerRole => {
    return {
      id: speaker.speaker_id, // Use speaker_id as the unique identifier
      speakerId: speaker.speaker_id,
      name: speaker.name,
      description: speaker.description,
      voiceFilename: speaker.voice_filename,
      voiceFile: null, // File uploads handled separately
      createdAt: speaker.created_at,
      updatedAt: speaker.updated_at,
    };
  };

  // Load speaker roles from backend on mount and when projectId changes
  useEffect(() => {
    const loadSpeakers = async () => {
      if (!projectId) return;

      setLoading(true);
      setError(null);

      try {
        const response = await api.listSpeakers(projectId);
        const roles = response.speakers.map(backendToFrontend);
        setSpeakerRoles(roles);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "Failed to load speakers";
        setError(errorMessage);
        console.error("Error loading speakers:", err);
      } finally {
        setLoading(false);
      }
    };

    loadSpeakers();
  }, [projectId]);

  const addSpeakerRole = async (): Promise<void> => {
    setError(null);

    // Create a local-only speaker role (not synced to backend yet)
    // It will be created on the backend when the user uploads a voice file
    const newRole: SpeakerRole = {
      id: `temp-${Date.now()}`, // Temporary ID
      speakerId: `Speaker ${speakerRoles.length + 1}`,
      name: `Speaker ${speakerRoles.length + 1}`,
      description: "",
      voiceFilename: null,
      voiceFile: null,
    };

    setSpeakerRoles([...speakerRoles, newRole]);
  };

  const updateSpeakerRole = async (speakerId: string, updates: Partial<SpeakerRole>): Promise<void> => {
    setError(null);

    // Update local state immediately for better UX
    setSpeakerRoles(roles =>
      roles.map(role =>
        role.speakerId === speakerId ? { ...role, ...updates } : role
      )
    );

    // Check if this is a temporary speaker
    const role = speakerRoles.find(r => r.speakerId === speakerId);
    if (!role) {
      throw new Error("Speaker not found");
    }

    // Only sync to backend if it's not a temporary speaker and name or description changed
    if (!role.id.startsWith('temp-') && (updates.name !== undefined || updates.description !== undefined)) {
      try {
        const updateData: { name?: string; description?: string } = {};
        if (updates.name !== undefined) updateData.name = updates.name;
        if (updates.description !== undefined) updateData.description = updates.description;

        const updatedSpeaker = await api.updateSpeaker(projectId, speakerId, updateData);

        // Update with backend response
        setSpeakerRoles(roles =>
          roles.map(role =>
            role.speakerId === speakerId ? backendToFrontend(updatedSpeaker) : role
          )
        );
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : "Failed to update speaker";
        setError(errorMessage);
        throw err;
      }
    }
  };

  const deleteSpeakerRole = async (speakerId: string): Promise<void> => {
    setLoading(true);
    setError(null);

    try {
      // Find the speaker to check if it's temporary
      const role = speakerRoles.find(r => r.speakerId === speakerId);
      if (!role) {
        throw new Error("Speaker not found");
      }

      // If it's a temporary speaker, just remove it from local state
      if (role.id.startsWith('temp-')) {
        setSpeakerRoles(speakerRoles.filter(r => r.speakerId !== speakerId));
      } else {
        // Delete from backend
        await api.deleteSpeaker(projectId, speakerId);

        // Reload all speakers to get updated speaker_ids (backend auto-reindexes)
        const response = await api.listSpeakers(projectId);
        const roles = response.speakers.map(backendToFrontend);
        setSpeakerRoles(roles);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to delete speaker";
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const uploadVoiceFile = async (speakerId: string, file: File): Promise<void> => {
    setLoading(true);
    setError(null);

    try {
      // Get current speaker data
      const currentRole = speakerRoles.find(r => r.speakerId === speakerId);
      if (!currentRole) {
        throw new Error("Speaker not found");
      }

      // Check if this is a temporary speaker (not yet created on backend)
      const isTemporary = currentRole.id.startsWith('temp-');

      if (isTemporary) {
        // Create a new speaker on the backend
        await api.createSpeaker(projectId, {
          name: currentRole.name,
          description: currentRole.description,
          voice_file: file,
        });

        // Reload all speakers to get the newly created speaker with proper ID
        const response = await api.listSpeakers(projectId);
        const roles = response.speakers.map(backendToFrontend);
        setSpeakerRoles(roles);
      } else {
        // Update existing speaker's voice file (keeps speaker ID)
        const updatedSpeaker = await api.updateVoiceFile(projectId, speakerId, file);

        // Update in local state
        setSpeakerRoles(roles =>
          roles.map(role =>
            role.speakerId === speakerId ? backendToFrontend(updatedSpeaker) : role
          )
        );
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to upload voice file";
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return (
    <SpeakerRoleContext.Provider
      value={{
        speakerRoles,
        addSpeakerRole,
        updateSpeakerRole,
        deleteSpeakerRole,
        uploadVoiceFile,
        hasUnsavedChanges,
        loading,
        error,
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
