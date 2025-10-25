'use client';

import React from 'react';
import { useGeneration } from '@/lib/GenerationContext';
import { useProject } from '@/lib/ProjectContext';
import { api } from '@/lib/api';
import { InferencePhase } from '@/types/generation';

export default function CurrentGeneration() {
  const { currentGeneration } = useGeneration();
  const { currentProject } = useProject();

  if (!currentGeneration) {
    return null;
  }

  const getStatusColor = (status: InferencePhase): string => {
    switch (status) {
      case InferencePhase.COMPLETED:
        return 'bg-green-100 text-green-800 border-green-300';
      case InferencePhase.FAILED:
        return 'bg-red-100 text-red-800 border-red-300';
      case InferencePhase.PENDING:
        return 'bg-gray-100 text-gray-800 border-gray-300';
      case InferencePhase.PREPROCESSING:
      case InferencePhase.INFERENCING:
      case InferencePhase.SAVING_AUDIO:
        return 'bg-blue-100 text-blue-800 border-blue-300';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  };

  const getStatusLabel = (status: InferencePhase): string => {
    switch (status) {
      case InferencePhase.COMPLETED:
        return 'Completed';
      case InferencePhase.FAILED:
        return 'Failed';
      case InferencePhase.PENDING:
        return 'Pending';
      case InferencePhase.PREPROCESSING:
        return 'Preprocessing';
      case InferencePhase.INFERENCING:
        return 'Inferencing';
      case InferencePhase.SAVING_AUDIO:
        return 'Saving Audio';
      default:
        return status;
    }
  };

  const isActive = ['pending', 'preprocessing', 'inferencing', 'saving_audio'].includes(
    currentGeneration.status
  );

  // Extract progress info from details if available
  const currentStep = currentGeneration.details?.current || null;
  const totalSteps = currentGeneration.details?.total_step || null;
  const progressPercentage = currentStep && totalSteps
    ? (currentStep / totalSteps) * 100
    : currentGeneration.percentage;

  // Render phase-specific details
  const renderPhaseDetails = () => {
    const details = currentGeneration.details;

    // PREPROCESSING phase
    if (currentGeneration.status === InferencePhase.PREPROCESSING && details) {
      return (
        <div className="space-y-3">
          <label className="text-sm font-medium opacity-75 block">Preprocessing Information</label>

          {details.unique_speaker_names && (
            <div className="bg-white bg-opacity-50 rounded p-3">
              <p className="text-xs font-medium mb-1">Speakers ({details.unique_speaker_names.length})</p>
              <div className="flex flex-wrap gap-1">
                {details.unique_speaker_names.map((speaker: string, idx: number) => (
                  <span key={idx} className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">
                    {speaker}
                  </span>
                ))}
              </div>
            </div>
          )}

          {details.scripts && (
            <div className="bg-white bg-opacity-50 rounded p-3">
              <p className="text-xs font-medium mb-1">Dialog Lines: {details.scripts.length}</p>
            </div>
          )}
        </div>
      );
    }

    // SAVING_AUDIO or COMPLETED phase
    if ((currentGeneration.status === InferencePhase.SAVING_AUDIO ||
         currentGeneration.status === InferencePhase.COMPLETED) && details) {
      const audioUrl = currentProject && currentGeneration.output_filename
        ? api.getGenerationDownloadUrl(currentProject.id, currentGeneration.request_id)
        : null;

      return (
        <div className="space-y-3">
          {/* Audio Player (for completed generations) */}
          {currentGeneration.status === InferencePhase.COMPLETED && audioUrl && (
            <div className="bg-white bg-opacity-90 rounded-lg p-4 border-2 border-white">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M18 3a1 1 0 00-1.196-.98l-10 2A1 1 0 006 5v9.114A4.369 4.369 0 005 14c-1.657 0-3 .895-3 2s1.343 2 3 2 3-.895 3-2V7.82l8-1.6v5.894A4.37 4.37 0 0015 12c-1.657 0-3 .895-3 2s1.343 2 3 2 3-.895 3-2V3z" />
                  </svg>
                  <label className="text-sm font-semibold">Generated Audio</label>
                </div>
                <a
                  href={audioUrl + '?download=true'}
                  download={currentGeneration.output_filename}
                  className="px-3 py-1 text-xs bg-current text-white rounded-lg hover:opacity-80 transition-opacity flex items-center gap-1"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  Download
                </a>
              </div>
              <audio
                controls
                className="w-full"
                preload="metadata"
              >
                <source src={audioUrl} type="audio/wav" />
                Your browser does not support the audio element.
              </audio>
              <div className="mt-2 flex items-center gap-4 text-xs opacity-75">
                {details.audio_duration_seconds && (
                  <span>Duration: {details.audio_duration_seconds.toFixed(2)}s</span>
                )}
                {currentGeneration.output_filename && (
                  <span className="font-mono truncate">{currentGeneration.output_filename}</span>
                )}
              </div>
            </div>
          )}

          <label className="text-sm font-medium opacity-75 block">Generation Statistics</label>

          {/* Performance Summary */}
          {(details.generation_time !== undefined || details.audio_duration_seconds !== undefined || details.real_time_factor !== undefined) && (
            <div className="bg-white bg-opacity-90 rounded p-4">
              <p className="text-xs font-medium mb-3 opacity-75">⚡ Performance Metrics</p>
              <div className="grid grid-cols-3 gap-3">
                {details.generation_time !== undefined && (
                  <div className="text-center">
                    <p className="text-2xl font-bold">{details.generation_time.toFixed(2)}s</p>
                    <p className="text-xs opacity-75 mt-1">Generation Time</p>
                  </div>
                )}
                {details.audio_duration_seconds !== undefined && (
                  <div className="text-center">
                    <p className="text-2xl font-bold">{details.audio_duration_seconds.toFixed(2)}s</p>
                    <p className="text-xs opacity-75 mt-1">Audio Duration</p>
                  </div>
                )}
                {details.real_time_factor !== undefined && (
                  <div className="text-center">
                    <p className="text-2xl font-bold">{details.real_time_factor.toFixed(2)}×</p>
                    <p className="text-xs opacity-75 mt-1">Real-Time Factor</p>
                  </div>
                )}
              </div>
              {details.number_of_segments !== undefined && (
                <div className="mt-3 pt-3 border-t border-current border-opacity-20 text-center">
                  <p className="text-xs opacity-75">
                    Generated <span className="font-bold">{details.number_of_segments}</span> segments
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Token Statistics */}
          <div className="bg-white bg-opacity-90 rounded p-4">
            <p className="text-xs font-medium mb-3 opacity-75">🔢 Token Statistics</p>
            <div className="grid grid-cols-3 gap-3">
              {details.prefilling_tokens !== undefined && (
                <div className="text-center">
                  <p className="text-xl font-bold">{details.prefilling_tokens.toLocaleString()}</p>
                  <p className="text-xs opacity-75 mt-1">Prefilling</p>
                </div>
              )}
              {details.generated_tokens !== undefined && (
                <div className="text-center">
                  <p className="text-xl font-bold">{details.generated_tokens.toLocaleString()}</p>
                  <p className="text-xs opacity-75 mt-1">Generated</p>
                </div>
              )}
              {details.total_tokens !== undefined && (
                <div className="text-center">
                  <p className="text-xl font-bold">{details.total_tokens.toLocaleString()}</p>
                  <p className="text-xs opacity-75 mt-1">Total Tokens</p>
                </div>
              )}
            </div>
          </div>

          {/* Speaker Information */}
          {details.unique_speaker_names && (
            <div className="bg-white bg-opacity-90 rounded p-3">
              <p className="text-xs font-medium mb-2 opacity-75">🎤 Speakers Used ({details.unique_speaker_names.length})</p>
              <div className="flex flex-wrap gap-1">
                {details.unique_speaker_names.map((speaker: string, idx: number) => (
                  <span key={idx} className="px-3 py-1.5 bg-white rounded-full text-xs font-medium border border-current border-opacity-30">
                    {speaker}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Output File Path */}
          {details.output_audio_path && (
            <div className="bg-white bg-opacity-90 rounded p-3">
              <p className="text-xs font-medium mb-1 opacity-75">📁 Output Path</p>
              <p className="text-xs font-mono break-all opacity-75">{details.output_audio_path}</p>
            </div>
          )}
        </div>
      );
    }

    // FAILED phase
    if (currentGeneration.status === InferencePhase.FAILED && details) {
      return (
        <div className="space-y-3">
          <label className="text-sm font-medium opacity-75 block">Error Information</label>
          <div className="bg-red-50 bg-opacity-50 rounded p-3">
            <pre className="text-xs whitespace-pre-wrap text-red-900">
              {details.error || JSON.stringify(details, null, 2)}
            </pre>
          </div>
        </div>
      );
    }

    // Fallback for any other details
    if (details && Object.keys(details).length > 0) {
      return (
        <div className="space-y-3">
          <label className="text-sm font-medium opacity-75 block">Additional Details</label>
          <div className="bg-white bg-opacity-50 rounded p-3 max-h-48 overflow-y-auto">
            <pre className="text-xs whitespace-pre-wrap">
              {JSON.stringify(details, null, 2)}
            </pre>
          </div>
        </div>
      );
    }

    return null;
  };

  return (
    <div className={`border-2 rounded-lg p-6 ${getStatusColor(currentGeneration.status)}`}>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">Current Generation</h2>
        {isActive && (
          <div className="flex items-center gap-2">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-current"></div>
            <span className="text-sm font-medium">Processing...</span>
          </div>
        )}
      </div>

      <div className="space-y-4">
        {/* Status */}
        <div>
          <label className="text-sm font-medium opacity-75">Status</label>
          <p className="text-lg font-semibold">{getStatusLabel(currentGeneration.status)}</p>
        </div>

        {/* Session ID */}
        <div>
          <label className="text-sm font-medium opacity-75">Session ID</label>
          <p className="text-sm">{currentGeneration.session_id}</p>
        </div>

        {/* Request ID */}
        <div>
          <label className="text-sm font-medium opacity-75">Request ID</label>
          <p className="text-sm font-mono">{currentGeneration.request_id}</p>
        </div>

        {/* Progress Bar (if inferencing) */}
        {isActive && progressPercentage !== null && (
          <div>
            <label className="text-sm font-medium opacity-75 mb-2 block">Progress</label>
            <div className="w-full bg-white bg-opacity-50 rounded-full h-4">
              <div
                className="bg-current h-4 rounded-full transition-all duration-300 flex items-center justify-center"
                style={{ width: `${Math.min(100, Math.max(0, progressPercentage))}%` }}
              >
                <span className="text-xs font-medium text-white px-2">
                  {progressPercentage.toFixed(1)}%
                </span>
              </div>
            </div>
            {currentStep !== null && totalSteps !== null && (
              <p className="text-xs mt-1 opacity-75">
                Step {currentStep} of {totalSteps}
              </p>
            )}
          </div>
        )}

        {/* Model Settings */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-sm font-medium opacity-75">Model Type</label>
            <p className="text-sm">{currentGeneration.model_dtype}</p>
          </div>
          <div>
            <label className="text-sm font-medium opacity-75">CFG Scale</label>
            <p className="text-sm">{currentGeneration.cfg_scale ?? 'N/A'}</p>
          </div>
          <div>
            <label className="text-sm font-medium opacity-75">Seeds</label>
            <p className="text-sm">{currentGeneration.seeds}</p>
          </div>
          <div>
            <label className="text-sm font-medium opacity-75">Attention</label>
            <p className="text-sm">{currentGeneration.attn_implementation ?? 'N/A'}</p>
          </div>
        </div>

        {/* Phase-Specific Details */}
        {renderPhaseDetails()}

        {/* Timestamps */}
        <div className="text-xs opacity-75 pt-2 border-t border-current border-opacity-20">
          <p>Created: {new Date(currentGeneration.created_at).toLocaleString()}</p>
          <p>Updated: {new Date(currentGeneration.updated_at).toLocaleString()}</p>
        </div>
      </div>
    </div>
  );
}
