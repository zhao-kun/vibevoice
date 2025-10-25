'use client';

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
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

  // Fetch all generations for the project
  const fetchGenerations = useCallback(async () => {
    if (!projectId) return;

    try {
      setLoading(true);
      setError(null);
      const response = await api.listGenerations(projectId);
      setGenerations(response.generations);
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
      // Backend returns null if no active generation (this is normal, not an error)
      setCurrentGeneration(response.generation);
    } catch (err) {
      // Only log actual errors (network issues, server errors, etc.)
      console.error('Error fetching current generation:', err);
      setCurrentGeneration(null);
    }
  }, []);

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
    if (!currentGeneration) return;

    const isActive = ['pending', 'preprocessing', 'inferencing', 'saving_audio'].includes(
      currentGeneration.status
    );

    if (!isActive) {
      // If generation is complete/failed, refresh the history list once
      fetchGenerations();
      return;
    }

    const interval = setInterval(() => {
      fetchCurrentGeneration();
      // Also refresh history periodically to keep it up to date
      fetchGenerations();
    }, 2000);

    return () => clearInterval(interval);
  }, [currentGeneration, fetchCurrentGeneration, fetchGenerations]);

  const value: GenerationContextType = {
    generations,
    currentGeneration,
    loading,
    error,
    fetchGenerations,
    fetchCurrentGeneration,
    startGeneration,
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
