"use client";

import React, { useState } from "react";
import { useSpeakerRole } from "@/lib/SpeakerRoleContext";
import { useProject } from "@/lib/ProjectContext";
import { api } from "@/lib/api";
import AudioUploader from "./AudioUploader";
import AudioPlayer from "./AudioPlayer";
import toast from "react-hot-toast";

export default function SpeakerRoleManager() {
  const { currentProject } = useProject();
  const {
    speakerRoles,
    addSpeakerRole,
    updateSpeakerRole,
    deleteSpeakerRole,
    uploadVoiceFile,
    removeVoiceFile,
    hasUnsavedChanges,
    loading,
    error,
  } = useSpeakerRole();

  const [localError, setLocalError] = useState<string | null>(null);

  const handleAddSpeaker = async () => {
    setLocalError(null);
    try {
      await addSpeakerRole();
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : "Failed to add speaker");
    }
  };

  const handleDeleteRole = async (speakerId: string) => {
    if (speakerRoles.length === 1) {
      toast.error("Cannot delete the last speaker role. At least one speaker is required.");
      return;
    }

    if (confirm("Are you sure you want to delete this speaker role? This will also delete the voice file. This action cannot be undone.")) {
      setLocalError(null);
      try {
        await deleteSpeakerRole(speakerId);
      } catch (err) {
        setLocalError(err instanceof Error ? err.message : "Failed to delete speaker");
      }
    }
  };

  const handleUploadVoice = async (speakerId: string, file: File) => {
    setLocalError(null);
    try {
      await uploadVoiceFile(speakerId, file);
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : "Failed to upload voice file");
    }
  };

  const handleRemoveVoice = async (speakerId: string) => {
    if (confirm("Are you sure you want to remove this voice file?")) {
      setLocalError(null);
      try {
        await removeVoiceFile(speakerId);
      } catch (err) {
        setLocalError(err instanceof Error ? err.message : "Failed to remove voice file");
      }
    }
  };

  const handleUpdateDescription = async (speakerId: string, description: string) => {
    setLocalError(null);
    try {
      await updateSpeakerRole(speakerId, { description });
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : "Failed to update description");
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Error Display */}
      {(error || localError) && (
        <div className="flex-shrink-0 bg-red-50 border-b border-red-200 px-4 py-3">
          <p className="text-sm text-red-800">
            {error || localError}
          </p>
        </div>
      )}

      {/* Action Bar */}
      <div className="flex-shrink-0 border-b border-gray-200 bg-white px-6 py-3">
        <div className="flex items-center justify-between">
          <p className="text-sm text-gray-600">
            Speaker IDs are automatically assigned. Upload audio samples for each speaker.
          </p>
          <button
            onClick={handleAddSpeaker}
            disabled={loading}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? "Adding..." : "+ Add Speaker"}
          </button>
        </div>
      </div>

      {/* Speaker Roles List */}
      <div className="flex-1 overflow-y-auto p-6">
        <div className="space-y-6 max-w-4xl mx-auto">
          {speakerRoles.map((role, index) => (
            <div
              key={role.id}
              className="border border-gray-300 rounded-lg p-6 bg-white shadow-sm"
            >
              {/* Header */}
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="flex items-center justify-center w-10 h-10 bg-blue-100 text-blue-700 font-semibold rounded-full">
                    {index + 1}
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">
                      {role.speakerId}
                    </h3>
                    <p className="text-xs text-gray-500">Auto-assigned ID</p>
                  </div>
                </div>
                <button
                  onClick={() => handleDeleteRole(role.speakerId)}
                  disabled={loading}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-red-600 bg-red-50 hover:bg-red-100 rounded-lg transition-colors border border-red-200 hover:border-red-300 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                  Delete
                </button>
              </div>

              {/* Description */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Description
                  <span className="text-gray-500 font-normal ml-1">
                    (help identify this voice)
                  </span>
                </label>
                <textarea
                  value={role.description}
                  onChange={(e) =>
                    handleUpdateDescription(role.speakerId, e.target.value)
                  }
                  placeholder="e.g., Deep male voice, professional tone..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                  rows={2}
                  disabled={loading}
                />
              </div>

              {/* Voice File */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Voice Sample
                </label>
                {role.voiceFilename && currentProject ? (
                  <AudioPlayer
                    voiceFileUrl={api.getVoiceFileUrl(currentProject.id, role.speakerId)}
                    voiceFileName={role.voiceFilename}
                    onRemove={() => handleRemoveVoice(role.speakerId)}
                  />
                ) : (
                  <AudioUploader
                    onUpload={(file) => handleUploadVoice(role.speakerId, file)}
                  />
                )}
              </div>
            </div>
          ))}

          {speakerRoles.length === 0 && (
            <div className="text-center py-12 text-gray-500">
              <p className="text-lg mb-2">No speaker roles yet</p>
              <p className="text-sm">Click "Add Speaker" to create your first speaker role</p>
            </div>
          )}
        </div>
      </div>

      {/* Loading Indicator */}
      {loading && (
        <div className="flex-shrink-0 bg-blue-50 border-t border-blue-200 px-4 py-3">
          <p className="text-sm text-blue-800">
            Syncing with server...
          </p>
        </div>
      )}
    </div>
  );
}
