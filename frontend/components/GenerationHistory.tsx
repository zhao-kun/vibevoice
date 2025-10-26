'use client';

import React, { useState, useMemo, useCallback } from 'react';
import { useProject } from '@/lib/ProjectContext';
import { useGeneration } from '@/lib/GenerationContext';
import { api } from '@/lib/api';
import type { Generation } from '@/types/generation';
import { InferencePhase } from '@/types/generation';

function GenerationHistory() {
  const { currentProject } = useProject();
  const { generations, loading, deleteGeneration, batchDeleteGenerations } = useGeneration();
  const [expandedId, setExpandedId] = useState<string | null>(null);

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);

  // Selection state
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  // Confirmation dialog state
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<'single' | 'bulk'>('single');
  const [singleDeleteId, setSingleDeleteId] = useState<string | null>(null);

  const toggleDetails = useCallback((requestId: string) => {
    setExpandedId(prev => prev === requestId ? null : requestId);
  }, []);

  // Memoize pagination calculations
  const paginationData = useMemo(() => {
    const totalPages = Math.ceil(generations.length / itemsPerPage);
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    const paginatedGenerations = generations.slice(startIndex, endIndex);

    return {
      totalPages,
      startIndex,
      endIndex,
      paginatedGenerations
    };
  }, [generations, currentPage, itemsPerPage]);

  const { totalPages, startIndex, endIndex, paginatedGenerations } = paginationData;

  // Selection handlers
  const toggleSelection = useCallback((requestId: string) => {
    setSelectedIds(prev => {
      const newSelected = new Set(prev);
      if (newSelected.has(requestId)) {
        newSelected.delete(requestId);
      } else {
        newSelected.add(requestId);
      }
      return newSelected;
    });
  }, []);

  const toggleSelectAll = useCallback(() => {
    setSelectedIds(prev => {
      if (prev.size === paginatedGenerations.length) {
        return new Set();
      } else {
        return new Set(paginatedGenerations.map(g => g.request_id));
      }
    });
  }, [paginatedGenerations]);

  const isAllSelected = paginatedGenerations.length > 0 && selectedIds.size === paginatedGenerations.length;
  const isSomeSelected = selectedIds.size > 0 && selectedIds.size < paginatedGenerations.length;

  // Delete handlers
  const handleDeleteClick = useCallback((requestId: string) => {
    setSingleDeleteId(requestId);
    setDeleteTarget('single');
    setShowDeleteDialog(true);
  }, []);

  const handleBulkDeleteClick = useCallback(() => {
    setDeleteTarget('bulk');
    setShowDeleteDialog(true);
  }, []);

  const handleConfirmDelete = useCallback(async () => {
    try {
      if (deleteTarget === 'single' && singleDeleteId) {
        await deleteGeneration(singleDeleteId);
      } else if (deleteTarget === 'bulk') {
        await batchDeleteGenerations(Array.from(selectedIds));
        setSelectedIds(new Set());
      }
      setShowDeleteDialog(false);
      setSingleDeleteId(null);
    } catch (error) {
      console.error('Delete failed:', error);
      alert(error instanceof Error ? error.message : 'Failed to delete generation(s)');
    }
  }, [deleteTarget, singleDeleteId, selectedIds, deleteGeneration, batchDeleteGenerations]);

  const handleCancelDelete = useCallback(() => {
    setShowDeleteDialog(false);
    setSingleDeleteId(null);
  }, []);

  // Pagination handlers
  const goToPage = useCallback((page: number) => {
    setCurrentPage(Math.max(1, Math.min(page, totalPages)));
  }, [totalPages]);

  const handleItemsPerPageChange = useCallback((value: number) => {
    setItemsPerPage(value);
    setCurrentPage(1); // Reset to first page when changing items per page
  }, []);

  const handleDownload = useCallback((generation: Generation) => {
    if (!currentProject || !generation.output_filename) return;

    // Add download=true parameter to force download
    const url = api.getGenerationDownloadUrl(currentProject.id, generation.request_id) + '?download=true';
    window.open(url, '_blank');
  }, [currentProject]);

  const getStatusColor = (status: InferencePhase): string => {
    switch (status) {
      case InferencePhase.COMPLETED:
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

  const formatDate = (isoDate: string): string => {
    return new Date(isoDate).toLocaleString();
  };

  // Render generation details (similar to CurrentGeneration component)
  const renderGenerationDetails = (generation: Generation) => {
    const details = generation.details;
    const audioUrl = currentProject && generation.output_filename
      ? api.getGenerationDownloadUrl(currentProject.id, generation.request_id)
      : null;

    return (
      <div className="mt-4 pt-4 border-t border-gray-200 space-y-4">
        {/* Audio Player (for completed generations) */}
        {generation.status === InferencePhase.COMPLETED && audioUrl && (
          <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-lg p-4 border border-green-200">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <svg className="w-5 h-5 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M18 3a1 1 0 00-1.196-.98l-10 2A1 1 0 006 5v9.114A4.369 4.369 0 005 14c-1.657 0-3 .895-3 2s1.343 2 3 2 3-.895 3-2V7.82l8-1.6v5.894A4.37 4.37 0 0015 12c-1.657 0-3 .895-3 2s1.343 2 3 2 3-.895 3-2V3z" />
                </svg>
                <label className="text-sm font-semibold text-gray-800">Generated Audio</label>
              </div>
              <a
                href={audioUrl + '?download=true'}
                download={generation.output_filename}
                className="px-3 py-1 text-xs bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors flex items-center gap-1"
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
            <div className="mt-2 flex items-center gap-4 text-xs text-gray-600">
              {details?.audio_duration_seconds && (
                <span>Duration: {details.audio_duration_seconds.toFixed(2)}s</span>
              )}
              {generation.output_filename && (
                <span className="font-mono truncate">{generation.output_filename}</span>
              )}
            </div>
          </div>
        )}

        {/* Basic Information */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-xs font-medium text-gray-600">Request ID</label>
            <p className="text-sm font-mono text-gray-900 break-all">{generation.request_id}</p>
          </div>
          <div>
            <label className="text-xs font-medium text-gray-600">Session ID</label>
            <p className="text-sm text-gray-900">{generation.session_id}</p>
          </div>
        </div>

        {/* Model Parameters */}
        <div>
          <label className="text-xs font-medium text-gray-600 block mb-2">Model Parameters</label>
          <div className="grid grid-cols-2 gap-3 bg-gray-50 rounded p-3">
            <div>
              <p className="text-xs text-gray-600">CFG Scale</p>
              <p className="text-sm font-semibold">{generation.cfg_scale ?? 'N/A'}</p>
            </div>
            <div>
              <p className="text-xs text-gray-600">Seeds</p>
              <p className="text-sm font-semibold">{generation.seeds}</p>
            </div>
            <div>
              <p className="text-xs text-gray-600">Attention</p>
              <p className="text-sm font-semibold">{generation.attn_implementation ?? 'N/A'}</p>
            </div>
            {generation.output_filename && (
              <div>
                <p className="text-xs text-gray-600">Output File</p>
                <p className="text-sm font-semibold font-mono text-xs break-all">{generation.output_filename}</p>
              </div>
            )}
          </div>
        </div>

        {/* Phase-specific details */}
        {generation.status === InferencePhase.PREPROCESSING && details && (
          <div>
            <label className="text-xs font-medium text-gray-600 block mb-2">Preprocessing Information</label>
            {details.unique_speaker_names && (
              <div className="bg-blue-50 rounded p-3">
                <p className="text-xs font-medium mb-2">Speakers ({details.unique_speaker_names.length})</p>
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
              <div className="bg-blue-50 rounded p-3 mt-2">
                <p className="text-xs">Dialog Lines: <span className="font-semibold">{details.scripts.length}</span></p>
              </div>
            )}
          </div>
        )}

        {(generation.status == InferencePhase.SAVING_AUDIO || generation.status == InferencePhase.COMPLETED) && details && (
          <div className="space-y-3">
            <label className="text-sm font-semibold text-gray-800 block border-b border-gray-300 pb-2">📊 Generation Statistics</label>

            {/* Performance Summary Card */}
            {(details.generation_time !== undefined || details.audio_duration_seconds !== undefined || details.real_time_factor !== undefined) && (
              <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg p-4 border border-green-200">
                <p className="text-xs font-semibold text-green-800 mb-3">⚡ Performance Metrics</p>
                <div className="grid grid-cols-3 gap-4">
                  {details.generation_time !== undefined && (
                    <div className="text-center">
                      <p className="text-2xl font-bold text-green-700">{details.generation_time.toFixed(2)}s</p>
                      <p className="text-xs text-gray-600 mt-1">Generation Time</p>
                    </div>
                  )}
                  {details.audio_duration_seconds !== undefined && (
                    <div className="text-center">
                      <p className="text-2xl font-bold text-blue-700">{details.audio_duration_seconds.toFixed(2)}s</p>
                      <p className="text-xs text-gray-600 mt-1">Audio Duration</p>
                    </div>
                  )}
                  {details.real_time_factor !== undefined && (
                    <div className="text-center">
                      <p className="text-2xl font-bold text-purple-700">{details.real_time_factor.toFixed(2)}×</p>
                      <p className="text-xs text-gray-600 mt-1">Real-Time Factor</p>
                    </div>
                  )}
                </div>
                {details.number_of_segments !== undefined && (
                  <div className="mt-3 pt-3 border-t border-green-200 text-center">
                    <p className="text-xs text-gray-600">
                      Generated <span className="font-bold text-green-700">{details.number_of_segments}</span> segments
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Token Statistics */}
            {(details.prefilling_tokens !== undefined || details.generated_tokens !== undefined || details.total_tokens !== undefined) && (
              <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                <p className="text-xs font-semibold text-blue-800 mb-3">🔢 Token Statistics</p>
                <div className="grid grid-cols-3 gap-4">
                  {details.prefilling_tokens !== undefined && (
                    <div className="text-center">
                      <p className="text-xl font-bold text-blue-700">{details.prefilling_tokens.toLocaleString()}</p>
                      <p className="text-xs text-gray-600 mt-1">Prefilling</p>
                    </div>
                  )}
                  {details.generated_tokens !== undefined && (
                    <div className="text-center">
                      <p className="text-xl font-bold text-blue-700">{details.generated_tokens.toLocaleString()}</p>
                      <p className="text-xs text-gray-600 mt-1">Generated</p>
                    </div>
                  )}
                  {details.total_tokens !== undefined && (
                    <div className="text-center">
                      <p className="text-xl font-bold text-blue-700">{details.total_tokens.toLocaleString()}</p>
                      <p className="text-xs text-gray-600 mt-1">Total Tokens</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Speakers */}
            {details.unique_speaker_names && (
              <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
                <p className="text-xs font-semibold text-purple-800 mb-2">🎤 Speakers Used ({details.unique_speaker_names.length})</p>
                <div className="flex flex-wrap gap-2">
                  {details.unique_speaker_names.map((speaker: string, idx: number) => (
                    <span key={idx} className="px-3 py-1.5 bg-purple-100 text-purple-900 rounded-full text-xs font-medium">
                      {speaker}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Output Path */}
            {details.output_audio_path && (
              <div className="bg-gray-100 rounded-lg p-4 border border-gray-300">
                <p className="text-xs font-semibold text-gray-700 mb-2">📁 Output File Path</p>
                <p className="text-xs font-mono break-all text-gray-600 bg-white p-2 rounded border border-gray-200">{details.output_audio_path}</p>
              </div>
            )}
          </div>
        )}

        {generation.status === InferencePhase.FAILED && details && (
          <div>
            <label className="text-xs font-medium text-gray-600 block mb-2">Error Information</label>
            <div className="bg-red-50 rounded p-3">
              <pre className="text-xs whitespace-pre-wrap text-red-900">
                {details.error || JSON.stringify(details, null, 2)}
              </pre>
            </div>
          </div>
        )}

        {/* Timestamps */}
        <div className="bg-gray-50 rounded p-3">
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <p className="text-gray-600">Created</p>
              <p className="font-medium">{formatDate(generation.created_at)}</p>
            </div>
            <div>
              <p className="text-gray-600">Updated</p>
              <p className="font-medium">{formatDate(generation.updated_at)}</p>
            </div>
          </div>
        </div>
      </div>
    );
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
      {/* Header with selection controls */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-xl font-semibold">Generation History</h2>
          {selectedIds.size > 0 && (
            <button
              onClick={handleBulkDeleteClick}
              className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 flex items-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
              Delete Selected ({selectedIds.size})
            </button>
          )}
        </div>
        <p className="text-sm text-gray-600">
          Total: {generations.length} generation{generations.length !== 1 ? 's' : ''}
          {selectedIds.size > 0 && ` • ${selectedIds.size} selected`}
        </p>
      </div>

      {generations.length === 0 ? (
        <div className="flex items-center justify-center flex-1 text-gray-500">
          <p>No generations yet. Start your first generation!</p>
        </div>
      ) : (
        <>
          {/* Select all checkbox */}
          {paginatedGenerations.length > 0 && (
            <div className="mb-2 flex items-center gap-2 px-2">
              <input
                type="checkbox"
                checked={isAllSelected}
                ref={(el) => {
                  if (el) el.indeterminate = isSomeSelected;
                }}
                onChange={toggleSelectAll}
                className="w-4 h-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500"
              />
              <label className="text-sm text-gray-700 cursor-pointer" onClick={toggleSelectAll}>
                {isAllSelected ? 'Deselect All' : 'Select All on Page'}
              </label>
            </div>
          )}

          {/* Generation list */}
          <div className="space-y-2 overflow-y-auto flex-1">
            {paginatedGenerations.map((generation) => {
              const isExpanded = expandedId === generation.request_id;
              const isSelected = selectedIds.has(generation.request_id);

              return (
                <div
                  key={generation.request_id}
                  className={`border rounded-lg bg-white transition-all ${
                    isSelected ? 'border-blue-500 ring-2 ring-blue-200' : 'border-gray-300'
                  }`}
                >
                  <div className="p-4">
                    <div className="flex items-start gap-3">
                      {/* Selection checkbox */}
                      <div className="pt-1">
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => toggleSelection(generation.request_id)}
                          className="w-4 h-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500"
                          onClick={(e) => e.stopPropagation()}
                        />
                      </div>

                      {/* Main content */}
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

                      {/* Action buttons */}
                      <div className="flex flex-col gap-2">
                        <button
                          onClick={() => toggleDetails(generation.request_id)}
                          className="px-3 py-1 text-sm bg-gray-200 text-gray-700 rounded hover:bg-gray-300 flex items-center gap-1"
                        >
                          {isExpanded ? (
                            <>
                              <span>Hide</span>
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                              </svg>
                            </>
                          ) : (
                            <>
                              <span>Details</span>
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                              </svg>
                            </>
                          )}
                        </button>

                        {generation.status === InferencePhase.COMPLETED && generation.output_filename && (
                          <button
                            onClick={() => handleDownload(generation)}
                            className="px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600"
                          >
                            Download
                          </button>
                        )}

                        <button
                          onClick={() => handleDeleteClick(generation.request_id)}
                          className="px-3 py-1 text-sm bg-red-500 text-white rounded hover:bg-red-600 flex items-center gap-1"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                          Delete
                        </button>
                      </div>
                    </div>

                    {/* Expandable Details Section */}
                    {isExpanded && renderGenerationDetails(generation)}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Pagination controls */}
          <div className="mt-4 flex items-center justify-between border-t border-gray-300 pt-4">
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-700">Items per page:</label>
              <select
                value={itemsPerPage}
                onChange={(e) => handleItemsPerPageChange(Number(e.target.value))}
                className="border border-gray-300 rounded px-2 py-1 text-sm"
              >
                <option value={5}>5</option>
                <option value={10}>10</option>
                <option value={20}>20</option>
                <option value={50}>50</option>
              </select>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-700">
                Page {currentPage} of {totalPages} ({startIndex + 1}-{Math.min(endIndex, generations.length)} of {generations.length})
              </span>
              <div className="flex gap-1">
                <button
                  onClick={() => goToPage(1)}
                  disabled={currentPage === 1}
                  className="px-2 py-1 border border-gray-300 rounded text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100"
                >
                  ««
                </button>
                <button
                  onClick={() => goToPage(currentPage - 1)}
                  disabled={currentPage === 1}
                  className="px-2 py-1 border border-gray-300 rounded text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100"
                >
                  ‹
                </button>
                <button
                  onClick={() => goToPage(currentPage + 1)}
                  disabled={currentPage === totalPages}
                  className="px-2 py-1 border border-gray-300 rounded text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100"
                >
                  ›
                </button>
                <button
                  onClick={() => goToPage(totalPages)}
                  disabled={currentPage === totalPages}
                  className="px-2 py-1 border border-gray-300 rounded text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100"
                >
                  »»
                </button>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Confirmation Dialog */}
      {showDeleteDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold mb-2">Confirm Deletion</h3>
            <p className="text-gray-600 mb-4">
              {deleteTarget === 'single'
                ? 'Are you sure you want to delete this generation? This will also delete the generated audio file.'
                : `Are you sure you want to delete ${selectedIds.size} generation${selectedIds.size > 1 ? 's' : ''}? This will also delete the generated audio files.`}
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

// Export with React.memo to prevent unnecessary re-renders
export default React.memo(GenerationHistory);
