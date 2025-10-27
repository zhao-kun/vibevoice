"use client";

import React, { useState, useEffect } from "react";
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
    loading,
    error,
  } = useSpeakerRole();

  const [localError, setLocalError] = useState<string | null>(null);

  // Track local edits for descriptions
  const [localDescriptions, setLocalDescriptions] = useState<Record<string, string>>({});
  // Track saving state per speaker
  const [savingStates, setSavingStates] = useState<Record<string, boolean>>({});
  // Delete confirmation dialog state
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [deleteTargetId, setDeleteTargetId] = useState<string | null>(null);

  // Initialize local descriptions from speaker roles
  useEffect(() => {
    const descriptions: Record<string, string> = {};
    speakerRoles.forEach(role => {
      descriptions[role.speakerId] = role.description;
    });
    setLocalDescriptions(descriptions);
  }, [speakerRoles]);

  const handleAddSpeaker = async () => {
    setLocalError(null);
    try {
      await addSpeakerRole();
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : "Failed to add speaker");
    }
  };

  const handleDeleteClick = (speakerId: string) => {
    if (speakerRoles.length === 1) {
      toast.error("Cannot delete the last speaker role. At least one speaker is required.");
      return;
    }

    setDeleteTargetId(speakerId);
    setShowDeleteDialog(true);
  };

  const handleConfirmDelete = async () => {
    if (!deleteTargetId) return;

    setLocalError(null);
    try {
      await deleteSpeakerRole(deleteTargetId);
      toast.success("Speaker deleted successfully");
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : "Failed to delete speaker");
      toast.error(err instanceof Error ? err.message : "Failed to delete speaker");
    } finally {
      setShowDeleteDialog(false);
      setDeleteTargetId(null);
    }
  };

  const handleCancelDelete = () => {
    setShowDeleteDialog(false);
    setDeleteTargetId(null);
  };

  const handleUploadVoice = async (speakerId: string, file: File) => {
    setLocalError(null);
    try {
      await uploadVoiceFile(speakerId, file);
      toast.success("Voice file uploaded successfully");
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : "Failed to upload voice file");
      toast.error(err instanceof Error ? err.message : "Failed to upload voice file");
    }
  };

  const handleUpdateDescription = (speakerId: string, description: string) => {
    // Update local state only (no API call)
    setLocalDescriptions(prev => ({
      ...prev,
      [speakerId]: description,
    }));
  };

  const handleSaveSpeaker = async (speakerId: string) => {
    const localDesc = localDescriptions[speakerId] || "";
    const currentRole = speakerRoles.find(r => r.speakerId === speakerId);

    if (!currentRole || localDesc === currentRole.description) {
      return; // No changes to save
    }

    // Set saving state for this speaker
    setSavingStates(prev => ({ ...prev, [speakerId]: true }));
    setLocalError(null);

    try {
      await updateSpeakerRole(speakerId, { description: localDesc });
      toast.success(`${speakerId} saved successfully`);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to save changes";
      setLocalError(errorMessage);
      toast.error(errorMessage);
    } finally {
      setSavingStates(prev => ({ ...prev, [speakerId]: false }));
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
                  onClick={() => handleDeleteClick(role.speakerId)}
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
                  value={localDescriptions[role.speakerId] || ""}
                  onChange={(e) =>
                    handleUpdateDescription(role.speakerId, e.target.value)
                  }
                  placeholder="e.g., Deep male voice, professional tone..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                  rows={2}
                  disabled={loading || savingStates[role.speakerId]}
                />
                {/* Save Button */}
                {localDescriptions[role.speakerId] !== role.description && (
                  <div className="mt-2 flex items-center gap-2">
                    <button
                      onClick={() => handleSaveSpeaker(role.speakerId)}
                      disabled={savingStates[role.speakerId] || loading}
                      className="px-3 py-1.5 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1.5"
                    >
                      {savingStates[role.speakerId] ? (
                        <>
                          <svg className="animate-spin h-3.5 w-3.5" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                          </svg>
                          Saving...
                        </>
                      ) : (
                        <>
                          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          </svg>
                          Save Changes
                        </>
                      )}
                    </button>
                    <span className="text-xs text-orange-600">Unsaved changes</span>
                  </div>
                )}
              </div>

              {/* Voice File */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Voice Sample
                </label>
                {role.voiceFilename && currentProject ? (
                  <AudioPlayer
                    key={role.updatedAt || role.voiceFilename}
                    voiceFileUrl={`${api.getVoiceFileUrl(currentProject.id, role.speakerId)}?t=${role.updatedAt || Date.now()}`}
                    voiceFileName={role.voiceFilename}
                    onChangeVoice={(file) => handleUploadVoice(role.speakerId, file)}
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
              <p className="text-sm">Click &ldquo;Add Speaker&rdquo; to create your first speaker role</p>
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

      {/* Confirmation Dialog */}
      {showDeleteDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold mb-2">Confirm Deletion</h3>
            <p className="text-gray-600 mb-4">
              Are you sure you want to delete this speaker role? This will also delete the voice file and subsequent speakers will be reindexed. This action cannot be undone.
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={handleCancelDelete}
                className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-100"
              >
                Cancel
              </button>
              <button
                onClick={handleConfirmDelete}
                className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
