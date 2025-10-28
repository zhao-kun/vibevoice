'use client';

import React, { useState } from 'react';
import { useSession } from '@/lib/SessionContext';
import { useGeneration } from '@/lib/GenerationContext';
import type { CreateGenerationRequest } from '@/types/generation';

export default function GenerationForm() {
  const { sessions } = useSession();
  const { startGeneration, loading } = useGeneration();

  const [formData, setFormData] = useState<CreateGenerationRequest>({
    dialog_session_id: '',
    seeds: 42,
    cfg_scale: 1.3,
    model_dtype: 'float8_e4m3fn',
    attn_implementation: 'sdpa'
  });

  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);

    // Validation
    if (!formData.dialog_session_id) {
      setError('Please select a dialog session');
      return;
    }

    if (!formData.seeds || formData.seeds < 0) {
      setError('Seeds must be a positive number');
      return;
    }

    if (!formData.cfg_scale || formData.cfg_scale <= 0) {
      setError('CFG Scale must be greater than 0');
      return;
    }

    try {
      const generation = await startGeneration(formData);
      setSuccess(`Generation started successfully! Request ID: ${generation.request_id}`);

      // Reset form after 3 seconds
      setTimeout(() => {
        setSuccess(null);
      }, 3000);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to start generation';
      setError(errorMessage);
    }
  };

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target;

    // Convert numeric fields
    if (name === 'seeds') {
      setFormData(prev => ({ ...prev, [name]: parseInt(value) || 0 }));
    } else if (name === 'cfg_scale') {
      setFormData(prev => ({ ...prev, [name]: parseFloat(value) || 0 }));
    } else {
      setFormData(prev => ({ ...prev, [name]: value }));
    }
  };

  const generateRandomSeed = () => {
    const randomSeed = Math.floor(Math.random() * 1000000);
    setFormData(prev => ({ ...prev, seeds: randomSeed }));
  };

  return (
    <div className="border border-gray-300 rounded-lg p-6 bg-white">
      <h2 className="text-xl font-semibold mb-4">Start New Generation</h2>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Dialog Session Selection */}
        <div>
          <label htmlFor="dialog_session_id" className="block text-sm font-medium text-gray-700 mb-1">
            Dialog Session <span className="text-red-500">*</span>
          </label>
          <select
            id="dialog_session_id"
            name="dialog_session_id"
            value={formData.dialog_session_id}
            onChange={handleChange}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          >
            <option value="">Select a dialog session</option>
            {sessions.map(session => (
              <option key={session.sessionId} value={session.sessionId}>
                {session.name} {session.description && `- ${session.description}`}
              </option>
            ))}
          </select>
          {sessions.length === 0 && (
            <p className="text-xs text-gray-500 mt-1">
              No dialog sessions available. Create one first in Voice Contents Editor.
            </p>
          )}
        </div>

        {/* Model Type Selection */}
        <div>
          <label htmlFor="model_dtype" className="block text-sm font-medium text-gray-700 mb-1">
            Model Type <span className="text-red-500">*</span>
          </label>
          <select
            id="model_dtype"
            name="model_dtype"
            value={formData.model_dtype}
            onChange={handleChange}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          >
            <option value="float8_e4m3fn">float8_e4m3fn (Recommended)</option>
            <option value="bf16">bf16</option>
          </select>
          <p className="text-xs text-gray-500 mt-1">
            float8_e4m3fn is optimized for performance with minimal quality loss
          </p>
        </div>

        {/* CFG Scale */}
        <div>
          <label htmlFor="cfg_scale" className="block text-sm font-medium text-gray-700 mb-1">
            CFG Scale <span className="text-red-500">*</span>
          </label>
          <input
            type="number"
            id="cfg_scale"
            name="cfg_scale"
            value={formData.cfg_scale}
            onChange={handleChange}
            step="0.1"
            min="0.1"
            max="10"
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          />
          <p className="text-xs text-gray-500 mt-1">
            Classifier-Free Guidance scale (1.0-3.0 recommended, default: 1.3)
          </p>
        </div>

        {/* Seeds */}
        <div>
          <label htmlFor="seeds" className="block text-sm font-medium text-gray-700 mb-1">
            Random Seed <span className="text-red-500">*</span>
          </label>
          <div className="flex gap-2">
            <input
              type="number"
              id="seeds"
              name="seeds"
              value={formData.seeds}
              onChange={handleChange}
              min="0"
              className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
            />
            <button
              type="button"
              onClick={generateRandomSeed}
              className="px-3 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
              title="Generate random seed"
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 512 512"
                className="w-5 h-5"
              >
                <path
                  d="M302.87 255.5a47.37 47.37 0 1 1-47.37-47.37 47.37 47.37 0 0 1 47.37 47.37zM128.5 81.18a47.37 47.37 0 1 0 47.41 47.32 47.37 47.37 0 0 0-47.41-47.32zm253.91 0a47.37 47.37 0 1 0 47.41 47.32 47.37 47.37 0 0 0-47.32-47.32zM128.5 335.09a47.37 47.37 0 1 0 47.41 47.41 47.37 47.37 0 0 0-47.41-47.41zm253.91 0a47.37 47.37 0 1 0 47.41 47.41 47.37 47.37 0 0 0-47.32-47.41zm102 92.93a56.48 56.48 0 0 1-56.39 56.48h-344a56.48 56.48 0 0 1-56.52-56.48v-344A56.48 56.48 0 0 1 83.98 27.5h344a56.48 56.48 0 0 1 56.52 56.48zm-20-344a36.48 36.48 0 0 0-36.39-36.52h-344A36.48 36.48 0 0 0 47.5 83.98v344a36.48 36.48 0 0 0 36.48 36.52h344a36.48 36.48 0 0 0 36.52-36.48z"
                  fill="currentColor"
                />
              </svg>
            </button>
          </div>
          <p className="text-xs text-gray-500 mt-1">
            Random seed for reproducible generation (default: 42)
          </p>
        </div>

        {/* Attention Implementation (Fixed) */}
        <div>
          <label htmlFor="attn_implementation" className="block text-sm font-medium text-gray-700 mb-1">
            Attention Implementation
          </label>
          <input
            type="text"
            id="attn_implementation"
            name="attn_implementation"
            value={formData.attn_implementation}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg bg-gray-100 cursor-not-allowed"
            disabled
          />
          <p className="text-xs text-gray-500 mt-1">
            Fixed to SDPA (Scaled Dot Product Attention)
          </p>
        </div>

        {/* Error Message */}
        {error && (
          <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-sm text-red-800">{error}</p>
          </div>
        )}

        {/* Success Message */}
        {success && (
          <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
            <p className="text-sm text-green-800">{success}</p>
          </div>
        )}

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading || sessions.length === 0}
          className="w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? 'Starting Generation...' : 'Start Generation'}
        </button>
      </form>
    </div>
  );
}
