'use client';

import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import { api } from './api';
import type { Generation, CreateGenerationRequest } from '@/types/generation';

interface GenerationContextType {
  // State
  generations: Generation[];
  currentGeneration: Generation | null;
  loading: boolean;
  error: string | null;

  // Actions
  fetchGenerations: () => Promise<void>;
  fetchCurrentGeneration: () => Promise<void>;
  startGeneration: (request: CreateGenerationRequest) => Promise<Generation>;
  deleteGeneration: (requestId: string) => Promise<void>;
  batchDeleteGenerations: (requestIds: string[]) => Promise<{ deleted_count: number; failed_count: number }>;
  refreshAll: () => Promise<void>;
}

const GenerationContext = createContext<GenerationContextType | undefined>(undefined);

interface GenerationProviderProps {
  children: React.ReactNode;
  projectId: string;
}

export function GenerationProvider({ children, projectId }: GenerationProviderProps) {
  const [generations, setGenerations] = useState<Generation[]>([]);
  const [currentGeneration, setCurrentGeneration] = useState<Generation | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Track previous status to detect transitions
  const previousStatusRef = useRef<string | null>(null);
  // Track the request_id of the generation being polled
  const activeRequestIdRef = useRef<string | null>(null);

  // Fetch all generations for the project
  const fetchGenerations = useCallback(async () => {
    if (!projectId) return;

    try {
      setLoading(true);
      setError(null);
      const response = await api.listGenerations(projectId);
      // Sort by created_at in descending order (newest first)
      const sortedGenerations = response.generations.sort((a, b) => {
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      });
      setGenerations(sortedGenerations);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch generations';
      setError(errorMessage);
      console.error('Error fetching generations:', err);
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  // Fetch current generation status
  const fetchCurrentGeneration = useCallback(async () => {
    try {
      const response = await api.getCurrentGeneration();
      const newCurrentGeneration = response.generation;

      // Check if we had an active generation that just completed (became null)
      const hadActiveGeneration = previousStatusRef.current !== null && activeRequestIdRef.current !== null;
      const generationNowNull = newCurrentGeneration === null;

      // If we had an active generation and now it's null, it completed!
      if (hadActiveGeneration && generationNowNull) {
        const completedRequestId = activeRequestIdRef.current;

        // Do final poll to get the completed state
        if (projectId && completedRequestId) {
          try {
            const finalResponse = await api.getGeneration(projectId, completedRequestId);
            const finalGeneration = finalResponse.generation;

            // Update the generation in the list
            setGenerations(prevGenerations => {
              const index = prevGenerations.findIndex(g => g.request_id === completedRequestId);
              if (index !== -1) {
                const updated = [...prevGenerations];
                updated[index] = finalGeneration;
                return updated;
              }
              return [finalGeneration, ...prevGenerations];
            });

            // Refresh full history
            await fetchGenerations();
          } catch (err) {
            console.error('Error in final poll:', err);
            await fetchGenerations();
          }
        }

        // Clear tracking refs
        previousStatusRef.current = null;
        activeRequestIdRef.current = null;
      }

      // Backend returns null if no active generation (this is normal, not an error)
      setCurrentGeneration(newCurrentGeneration);

      // If there's a current generation, update it in the generations list without re-fetching
      if (newCurrentGeneration) {
        // Track the status for transition detection
        previousStatusRef.current = newCurrentGeneration.status;
        activeRequestIdRef.current = newCurrentGeneration.request_id;

        setGenerations(prevGenerations => {
          const index = prevGenerations.findIndex(g => g.request_id === newCurrentGeneration.request_id);
          if (index !== -1) {
            // Update existing generation in the list
            const updated = [...prevGenerations];
            updated[index] = newCurrentGeneration;
            return updated;
          }
          // If not found in list, add it at the beginning (newest first)
          return [newCurrentGeneration, ...prevGenerations];
        });
      }
    } catch (err) {
      // Only log actual errors (network issues, server errors, etc.)
      console.error('Error fetching current generation:', err);
      setCurrentGeneration(null);
    }
  }, [projectId, fetchGenerations]);

  // Start a new generation
  const startGeneration = useCallback(async (request: CreateGenerationRequest): Promise<Generation> => {
    if (!projectId) {
      throw new Error('No project selected');
    }

    try {
      setLoading(true);
      setError(null);
      const response = await api.createGeneration(projectId, request);

      // Update current generation
      setCurrentGeneration(response.generation);

      // Refresh generations list
      await fetchGenerations();

      return response.generation;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to start generation';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [projectId, fetchGenerations]);

  // Delete a single generation
  const deleteGeneration = useCallback(async (requestId: string): Promise<void> => {
    if (!projectId) {
      throw new Error('No project selected');
    }

    try {
      setLoading(true);
      setError(null);
      await api.deleteGeneration(projectId, requestId);

      // Refresh generations list
      await fetchGenerations();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to delete generation';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [projectId, fetchGenerations]);

  // Batch delete multiple generations
  const batchDeleteGenerations = useCallback(async (requestIds: string[]): Promise<{ deleted_count: number; failed_count: number }> => {
    if (!projectId) {
      throw new Error('No project selected');
    }

    try {
      setLoading(true);
      setError(null);
      const response = await api.batchDeleteGenerations(projectId, requestIds);

      // Refresh generations list
      await fetchGenerations();

      return {
        deleted_count: response.deleted_count,
        failed_count: response.failed_count
      };
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to delete generations';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [projectId, fetchGenerations]);

  // Refresh both current and all generations
  const refreshAll = useCallback(async () => {
    await Promise.all([
      fetchGenerations(),
      fetchCurrentGeneration()
    ]);
  }, [fetchGenerations, fetchCurrentGeneration]);

  // Initial load
  useEffect(() => {
    if (projectId) {
      refreshAll();
    }
  }, [projectId, refreshAll]);

  // Poll for current generation updates (every 2 seconds when there's an active generation)
  useEffect(() => {
    // Only poll if there's an active generation
    if (!currentGeneration) {
      return;
    }

    const currentStatus = currentGeneration.status;
    const isActive = ['pending', 'preprocessing', 'inferencing', 'saving_audio'].includes(currentStatus);

    if (!isActive) {
      return;
    }

    // Active generation - set up polling interval
    const interval = setInterval(() => {
      fetchCurrentGeneration();
    }, 2000);

    return () => clearInterval(interval);
  }, [currentGeneration, fetchCurrentGeneration]);

  const value: GenerationContextType = {
    generations,
    currentGeneration,
    loading,
    error,
    fetchGenerations,
    fetchCurrentGeneration,
    startGeneration,
    deleteGeneration,
    batchDeleteGenerations,
    refreshAll
  };

  return (
    <GenerationContext.Provider value={value}>
      {children}
    </GenerationContext.Provider>
  );
}

export function useGeneration() {
  const context = useContext(GenerationContext);
  if (context === undefined) {
    throw new Error('useGeneration must be used within a GenerationProvider');
  }
  return context;
}
