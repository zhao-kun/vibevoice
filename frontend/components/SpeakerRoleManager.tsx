"use client";

import React, { useState, useEffect } from "react";
import { useSpeakerRole } from "@/lib/SpeakerRoleContext";
import { useProject } from "@/lib/ProjectContext";
import { useLanguage } from "@/lib/i18n/LanguageContext";
import { api } from "@/lib/api";
import AudioUploader from "./AudioUploader";
import AudioPlayer from "./AudioPlayer";
import VoiceRecorder from "./VoiceRecorder";
import toast from "react-hot-toast";

export default function SpeakerRoleManager() {
  const { currentProject } = useProject();
  const { t } = useLanguage();
  const {
    speakerRoles,
    addSpeakerRole,
    updateSpeakerRole,
    deleteSpeakerRole,
    uploadVoiceFile,
    removeVoiceFile,
    trimAudio,
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
  // Track input method per speaker (upload or record)
  const [inputMethod, setInputMethod] = useState<Record<string, 'upload' | 'record'>>({});

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
      toast.error(t('speaker.deleteError'));
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
      toast.success(t('speaker.deleteSuccess'));
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : t('speaker.deleteError'));
      toast.error(err instanceof Error ? err.message : t('speaker.deleteError'));
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
      toast.success(t('speaker.uploadSuccess'));
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : t('speaker.uploadError'));
      toast.error(err instanceof Error ? err.message : t('speaker.uploadError'));
    }
  };

  const handleTrimAudio = async (speakerId: string, startTime: number, endTime: number) => {
    setLocalError(null);
    try {
      await trimAudio(speakerId, startTime, endTime);
      toast.success(t('speaker.updateSuccess'));
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : t('speaker.updateError'));
      toast.error(err instanceof Error ? err.message : t('speaker.updateError'));
    }
  };

  const handleRemoveVoice = async (speakerId: string) => {
    setLocalError(null);
    try {
      await removeVoiceFile(speakerId);
      toast.success(t('speaker.deleteSuccess'));
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : t('speaker.deleteError'));
      toast.error(err instanceof Error ? err.message : t('speaker.deleteError'));
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
      toast.success(t('speaker.updateSuccess'));
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : t('speaker.updateError');
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
            {t('speaker.title')}
          </p>
          <button
            onClick={handleAddSpeaker}
            disabled={loading}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? t('common.loading') : t('speaker.addSpeaker')}
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
                    <p className="text-xs text-gray-500">{t('speaker.speakerName')}</p>
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
                  {t('common.delete')}
                </button>
              </div>

              {/* Description */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {t('speaker.speakerDescription')}
                </label>
                <textarea
                  value={localDescriptions[role.speakerId] || ""}
                  onChange={(e) =>
                    handleUpdateDescription(role.speakerId, e.target.value)
                  }
                  placeholder={t('project.enterProjectDescription')}
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
                          {t('common.loading')}
                        </>
                      ) : (
                        <>
                          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                          </svg>
                          {t('common.save')}
                        </>
                      )}
                    </button>
                  </div>
                )}
              </div>

              {/* Voice File */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {t('speaker.voiceFile')}
                </label>
                {role.voiceFilename && currentProject ? (
                  <AudioPlayer
                    key={role.updatedAt || role.voiceFilename}
                    voiceFileUrl={`${api.getVoiceFileUrl(currentProject.id, role.speakerId)}?t=${role.updatedAt || Date.now()}`}
                    voiceFileName={role.voiceFilename}
                    onChangeVoice={(file) => handleUploadVoice(role.speakerId, file)}
                    onTrimAudio={(startTime, endTime) => handleTrimAudio(role.speakerId, startTime, endTime)}
                    onRemoveVoice={() => handleRemoveVoice(role.speakerId)}
                  />
                ) : (
                  <div>
                    {/* Tabs for Upload/Record */}
                    <div className="flex border-b border-gray-200 mb-4">
                      <button
                        onClick={() => setInputMethod(prev => ({ ...prev, [role.speakerId]: 'upload' }))}
                        className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                          (inputMethod[role.speakerId] || 'upload') === 'upload'
                            ? 'border-blue-600 text-blue-600'
                            : 'border-transparent text-gray-500 hover:text-gray-700'
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                          </svg>
                          {t('common.upload')}
                        </div>
                      </button>
                      <button
                        onClick={() => setInputMethod(prev => ({ ...prev, [role.speakerId]: 'record' }))}
                        className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                          inputMethod[role.speakerId] === 'record'
                            ? 'border-blue-600 text-blue-600'
                            : 'border-transparent text-gray-500 hover:text-gray-700'
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                          </svg>
                          {t('speaker.uploadVoice')}
                        </div>
                      </button>
                    </div>

                    {/* Content based on selected tab */}
                    {(inputMethod[role.speakerId] || 'upload') === 'upload' ? (
                      <AudioUploader
                        onUpload={(file) => handleUploadVoice(role.speakerId, file)}
                      />
                    ) : (
                      <VoiceRecorder
                        onSave={(file) => handleUploadVoice(role.speakerId, file)}
                      />
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}

          {speakerRoles.length === 0 && (
            <div className="text-center py-12 text-gray-500">
              <p className="text-lg mb-2">{t('speaker.noVoice')}</p>
              <p className="text-sm">{t('speaker.addSpeaker')}</p>
            </div>
          )}
        </div>
      </div>

      {/* Loading Indicator */}
      {loading && (
        <div className="flex-shrink-0 bg-blue-50 border-t border-blue-200 px-4 py-3">
          <p className="text-sm text-blue-800">
            {t('common.loading')}
          </p>
        </div>
      )}

      {/* Confirmation Dialog */}
      {showDeleteDialog && (
        <div className="fixed inset-0 bg-gray-900/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4 shadow-xl">
            <h3 className="text-lg font-semibold mb-2">{t('common.confirm')}</h3>
            <p className="text-gray-600 mb-4">
              {t('speaker.deleteConfirm')}
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={handleCancelDelete}
                className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-100"
              >
                {t('common.cancel')}
              </button>
              <button
                onClick={handleConfirmDelete}
                className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600"
              >
                {t('common.delete')}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
