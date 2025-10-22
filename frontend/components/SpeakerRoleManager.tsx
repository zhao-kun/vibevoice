"use client";

import React from "react";
import { useSpeakerRole } from "@/lib/SpeakerRoleContext";
import AudioUploader from "./AudioUploader";
import AudioPlayer from "./AudioPlayer";

export default function SpeakerRoleManager() {
  const {
    speakerRoles,
    addSpeakerRole,
    updateSpeakerRole,
    deleteSpeakerRole,
    uploadVoiceFile,
    removeVoiceFile,
    saveSpeakerRoles,
    hasUnsavedChanges,
  } = useSpeakerRole();

  const handleDeleteRole = (id: string) => {
    if (speakerRoles.length === 1) {
      alert("Cannot delete the last speaker role. At least one speaker is required.");
      return;
    }

    if (confirm("Are you sure you want to delete this speaker role? This action cannot be undone.")) {
      deleteSpeakerRole(id);
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex-shrink-0 border-b border-gray-200 bg-white p-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">Speaker Roles</h2>
            <p className="text-sm text-gray-600 mt-1">
              Manage speaker voices for your project. Speaker IDs are automatically assigned.
            </p>
          </div>
          <div className="flex gap-2">
            <button
              onClick={addSpeakerRole}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              + Add Speaker
            </button>
            <button
              onClick={saveSpeakerRoles}
              disabled={!hasUnsavedChanges}
              className={`px-4 py-2 rounded-lg transition-colors ${
                hasUnsavedChanges
                  ? "bg-green-600 text-white hover:bg-green-700"
                  : "bg-gray-300 text-gray-500 cursor-not-allowed"
              }`}
            >
              {hasUnsavedChanges ? "Save Changes" : "Saved"}
            </button>
          </div>
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
                  onClick={() => handleDeleteRole(role.id)}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-red-600 bg-red-50 hover:bg-red-100 rounded-lg transition-colors border border-red-200 hover:border-red-300"
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
                    updateSpeakerRole(role.id, { description: e.target.value })
                  }
                  placeholder="e.g., Deep male voice, professional tone..."
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                  rows={2}
                />
              </div>

              {/* Voice File */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Voice Sample
                </label>
                {role.voiceFile ? (
                  <AudioPlayer
                    voiceFile={role.voiceFile}
                    onRemove={() => removeVoiceFile(role.id)}
                  />
                ) : (
                  <AudioUploader
                    onUpload={(file) => uploadVoiceFile(role.id, file)}
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

      {/* Unsaved Changes Indicator */}
      {hasUnsavedChanges && (
        <div className="flex-shrink-0 bg-yellow-50 border-t border-yellow-200 px-4 py-3">
          <p className="text-sm text-yellow-800">
            You have unsaved changes. Click "Save Changes" to persist your modifications.
          </p>
        </div>
      )}
    </div>
  );
}
