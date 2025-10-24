'use client';

import React, { useState } from 'react';
import { useProject } from '@/lib/ProjectContext';
import { useGeneration } from '@/lib/GenerationContext';
import { api } from '@/lib/api';
import type { Generation } from '@/types/generation';
import { InferencePhase } from '@/types/generation';

export default function GenerationHistory() {
  const { currentProject } = useProject();
  const { generations, loading } = useGeneration();
  const [selectedGeneration, setSelectedGeneration] = useState<Generation | null>(null);
  const [showDetails, setShowDetails] = useState(false);

  const handleViewDetails = (generation: Generation) => {
    setSelectedGeneration(generation);
    setShowDetails(true);
  };

  const handleDownload = (generation: Generation) => {
    if (!currentProject || !generation.output_filename) return;

    const url = api.getGenerationDownloadUrl(currentProject.id, generation.request_id);
    window.open(url, '_blank');
  };

  const getStatusColor = (status: InferencePhase): string => {
    switch (status) {
      case InferencePhase.COMPLETE:
        return 'bg-green-100 text-green-800';
      case InferencePhase.FAILED:
        return 'bg-red-100 text-red-800';
      case InferencePhase.PENDING:
        return 'bg-gray-100 text-gray-800';
      case InferencePhase.PREPROCESSING:
      case InferencePhase.INFERENCING:
      case InferencePhase.SAVING_AUDIO:
        return 'bg-blue-100 text-blue-800';
      default:
        return 'bg-gray-100 text-gray-800';
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

  const formatDate = (isoDate: string): string => {
    return new Date(isoDate).toLocaleString();
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-gray-500">Loading generation history...</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <div className="mb-4">
        <h2 className="text-xl font-semibold">Generation History</h2>
        <p className="text-sm text-gray-600">
          Total: {generations.length} generation{generations.length !== 1 ? 's' : ''}
        </p>
      </div>

      {generations.length === 0 ? (
        <div className="flex items-center justify-center flex-1 text-gray-500">
          <p>No generations yet. Start your first generation!</p>
        </div>
      ) : (
        <div className="space-y-2 overflow-y-auto flex-1">
          {generations.map((generation) => (
            <div
              key={generation.request_id}
              className="border rounded-lg p-4 hover:bg-gray-50 transition-colors"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    <span
                      className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(
                        generation.status
                      )}`}
                    >
                      {getStatusLabel(generation.status)}
                    </span>
                    <span className="text-xs text-gray-500">
                      {generation.model_dtype}
                    </span>
                  </div>

                  <p className="text-sm text-gray-600 mb-1">
                    Session: <span className="font-medium">{generation.session_id}</span>
                  </p>

                  <p className="text-xs text-gray-500">
                    Created: {formatDate(generation.created_at)}
                  </p>

                  {generation.percentage !== null && (
                    <div className="mt-2">
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-blue-500 h-2 rounded-full transition-all"
                          style={{ width: `${generation.percentage}%` }}
                        />
                      </div>
                      <p className="text-xs text-gray-500 mt-1">
                        {generation.percentage.toFixed(1)}%
                      </p>
                    </div>
                  )}
                </div>

                <div className="flex flex-col gap-2 ml-4">
                  <button
                    onClick={() => handleViewDetails(generation)}
                    className="px-3 py-1 text-sm bg-gray-200 text-gray-700 rounded hover:bg-gray-300"
                  >
                    Details
                  </button>

                  {generation.status === InferencePhase.COMPLETE && generation.output_filename && (
                    <button
                      onClick={() => handleDownload(generation)}
                      className="px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600"
                    >
                      Download
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Details Modal */}
      {showDetails && selectedGeneration && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">Generation Details</h3>
              <button
                onClick={() => setShowDetails(false)}
                className="text-gray-500 hover:text-gray-700"
              >
                ✕
              </button>
            </div>

            <div className="space-y-3">
              <div>
                <label className="text-sm font-medium text-gray-700">Request ID</label>
                <p className="text-sm text-gray-900 font-mono">{selectedGeneration.request_id}</p>
              </div>

              <div>
                <label className="text-sm font-medium text-gray-700">Session ID</label>
                <p className="text-sm text-gray-900">{selectedGeneration.session_id}</p>
              </div>

              <div>
                <label className="text-sm font-medium text-gray-700">Status</label>
                <p className="text-sm text-gray-900">{getStatusLabel(selectedGeneration.status)}</p>
              </div>

              <div>
                <label className="text-sm font-medium text-gray-700">Model Type</label>
                <p className="text-sm text-gray-900">{selectedGeneration.model_dtype}</p>
              </div>

              <div>
                <label className="text-sm font-medium text-gray-700">CFG Scale</label>
                <p className="text-sm text-gray-900">{selectedGeneration.cfg_scale ?? 'N/A'}</p>
              </div>

              <div>
                <label className="text-sm font-medium text-gray-700">Seeds</label>
                <p className="text-sm text-gray-900">{selectedGeneration.seeds}</p>
              </div>

              <div>
                <label className="text-sm font-medium text-gray-700">Attention Implementation</label>
                <p className="text-sm text-gray-900">{selectedGeneration.attn_implementation ?? 'N/A'}</p>
              </div>

              {selectedGeneration.output_filename && (
                <div>
                  <label className="text-sm font-medium text-gray-700">Output Filename</label>
                  <p className="text-sm text-gray-900 font-mono">{selectedGeneration.output_filename}</p>
                </div>
              )}

              <div>
                <label className="text-sm font-medium text-gray-700">Created At</label>
                <p className="text-sm text-gray-900">{formatDate(selectedGeneration.created_at)}</p>
              </div>

              <div>
                <label className="text-sm font-medium text-gray-700">Updated At</label>
                <p className="text-sm text-gray-900">{formatDate(selectedGeneration.updated_at)}</p>
              </div>

              {Object.keys(selectedGeneration.details).length > 0 && (
                <div>
                  <label className="text-sm font-medium text-gray-700">Additional Details</label>
                  <pre className="mt-1 text-xs bg-gray-50 p-3 rounded overflow-x-auto">
                    {JSON.stringify(selectedGeneration.details, null, 2)}
                  </pre>
                </div>
              )}
            </div>

            <div className="mt-6 flex justify-end">
              <button
                onClick={() => setShowDetails(false)}
                className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
