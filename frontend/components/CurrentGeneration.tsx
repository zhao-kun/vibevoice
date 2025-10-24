'use client';

import React from 'react';
import { useGeneration } from '@/lib/GenerationContext';
import { InferencePhase } from '@/types/generation';

export default function CurrentGeneration() {
  const { currentGeneration } = useGeneration();

  if (!currentGeneration) {
    return null;
  }

  const getStatusColor = (status: InferencePhase): string => {
    switch (status) {
      case InferencePhase.COMPLETE:
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
      case InferencePhase.COMPLETE:
        return 'Complete';
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

        {/* Additional Details */}
        {Object.keys(currentGeneration.details).length > 0 && (
          <div>
            <label className="text-sm font-medium opacity-75 mb-2 block">Details</label>
            <div className="bg-white bg-opacity-50 rounded p-3 max-h-48 overflow-y-auto">
              <pre className="text-xs whitespace-pre-wrap">
                {JSON.stringify(currentGeneration.details, null, 2)}
              </pre>
            </div>
          </div>
        )}

        {/* Timestamps */}
        <div className="text-xs opacity-75 pt-2 border-t border-current border-opacity-20">
          <p>Created: {new Date(currentGeneration.created_at).toLocaleString()}</p>
          <p>Updated: {new Date(currentGeneration.updated_at).toLocaleString()}</p>
        </div>
      </div>
    </div>
  );
}
