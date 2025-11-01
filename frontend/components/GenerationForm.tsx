'use client';

import React, { useState } from 'react';
import { useSession } from '@/lib/SessionContext';
import { useGeneration } from '@/lib/GenerationContext';
import { useLanguage } from '@/lib/i18n/LanguageContext';
import type { CreateGenerationRequest, OffloadingMode, OffloadingPreset } from '@/types/generation';

// Preset information for offloading configurations
const PRESET_INFO = {
  balanced: {
    vram_savings: '~5GB',
    slowdown: '~2.0x',
    gpu_layers: 12,
    description: 'Recommended for 12GB+ GPUs (RTX 3060 12GB, 4070)',
  },
  aggressive: {
    vram_savings: '~6GB',
    slowdown: '~2.5x',
    gpu_layers: 8,
    description: 'Recommended for 8-12GB GPUs (RTX 3060 8GB, 4060 8GB)',
  },
  extreme: {
    vram_savings: '~7GB',
    slowdown: '~3.5x',
    gpu_layers: 4,
    description: 'Recommended for 6-8GB GPUs (minimum viable VRAM)',
  },
};

export default function GenerationForm() {
  const { sessions } = useSession();
  const { startGeneration, loading } = useGeneration();
  const { t } = useLanguage();

  const [formData, setFormData] = useState<CreateGenerationRequest>({
    dialog_session_id: '',
    seeds: 42,
    cfg_scale: 1.3,
    model_dtype: 'float8_e4m3fn',
    attn_implementation: 'sdpa'
  });

  // Offloading configuration state
  const [offloadingEnabled, setOffloadingEnabled] = useState(false);
  const [offloadingMode, setOffloadingMode] = useState<OffloadingMode>('preset');
  const [offloadingPreset, setOffloadingPreset] = useState<OffloadingPreset>('balanced');
  const [manualGpuLayers, setManualGpuLayers] = useState(20);

  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);

    // Validation
    if (!formData.dialog_session_id) {
      setError(t('generation.validationSelectSession'));
      return;
    }

    if (!formData.seeds || formData.seeds < 0) {
      setError(t('generation.validationSeedsPositive'));
      return;
    }

    if (!formData.cfg_scale || formData.cfg_scale <= 0) {
      setError(t('generation.validationCfgScalePositive'));
      return;
    }

    // Build offloading configuration
    const offloading = offloadingEnabled ? {
      enabled: true,
      mode: offloadingMode,
      ...(offloadingMode === 'preset'
        ? { preset: offloadingPreset }
        : { num_gpu_layers: manualGpuLayers }
      )
    } : undefined;

    try {
      const generation = await startGeneration({
        ...formData,
        offloading,
      });
      setSuccess(t('generation.startSuccess').replace('{requestId}', generation.request_id));

      // Reset form after 3 seconds
      setTimeout(() => {
        setSuccess(null);
      }, 3000);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : t('generation.startError');
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
      <h2 className="text-xl font-semibold mb-4">{t('generation.startNewGeneration')}</h2>

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Dialog Session Selection */}
        <div>
          <label htmlFor="dialog_session_id" className="block text-sm font-medium text-gray-700 mb-1">
            {t('generation.dialogSession')} <span className="text-red-500">*</span>
          </label>
          <select
            id="dialog_session_id"
            name="dialog_session_id"
            value={formData.dialog_session_id}
            onChange={handleChange}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          >
            <option value="">{t('generation.selectDialogSession')}</option>
            {sessions.map(session => (
              <option key={session.sessionId} value={session.sessionId}>
                {session.name} {session.description && `- ${session.description}`}
              </option>
            ))}
          </select>
          {sessions.length === 0 && (
            <p className="text-xs text-gray-500 mt-1">
              {t('generation.noSessionsAvailable')}
            </p>
          )}
        </div>

        {/* Model Type Selection */}
        <div>
          <label htmlFor="model_dtype" className="block text-sm font-medium text-gray-700 mb-1">
            {t('generation.modelType')} <span className="text-red-500">*</span>
          </label>
          <select
            id="model_dtype"
            name="model_dtype"
            value={formData.model_dtype}
            onChange={handleChange}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          >
            <option value="float8_e4m3fn">{t('generation.float8Recommended')}</option>
            <option value="bf16">bf16</option>
          </select>
          <p className="text-xs text-gray-500 mt-1">
            {t('generation.float8Description')}
          </p>
        </div>

        {/* CFG Scale */}
        <div>
          <label htmlFor="cfg_scale" className="block text-sm font-medium text-gray-700 mb-1">
            {t('generation.cfgScale')} <span className="text-red-500">*</span>
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
            {t('generation.cfgScaleDescription')}
          </p>
        </div>

        {/* Seeds */}
        <div>
          <label htmlFor="seeds" className="block text-sm font-medium text-gray-700 mb-1">
            {t('generation.randomSeed')} <span className="text-red-500">*</span>
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
              title={t('generation.generateRandomSeed')}
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
            {t('generation.randomSeedDescription')}
          </p>
        </div>

        {/* Attention Implementation (Fixed) */}
        <div>
          <label htmlFor="attn_implementation" className="block text-sm font-medium text-gray-700 mb-1">
            {t('generation.attentionImplementation')}
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
            {t('generation.attentionFixed')}
          </p>
        </div>

        {/* Offloading Configuration Section */}
        <div className="border border-gray-300 rounded-lg p-4 bg-gray-50">
          <div className="flex items-center gap-2 mb-3">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
              <path fillRule="evenodd" d="M7.84 1.804A1 1 0 018.82 1h2.36a1 1 0 01.98.804l.331 1.652a6.993 6.993 0 011.929 1.115l1.598-.54a1 1 0 011.186.447l1.18 2.044a1 1 0 01-.205 1.251l-1.267 1.113a7.047 7.047 0 010 2.228l1.267 1.113a1 1 0 01.206 1.25l-1.18 2.045a1 1 0 01-1.187.447l-1.598-.54a6.993 6.993 0 01-1.929 1.115l-.33 1.652a1 1 0 01-.98.804H8.82a1 1 0 01-.98-.804l-.331-1.652a6.993 6.993 0 01-1.929-1.115l-1.598.54a1 1 0 01-1.186-.447l-1.18-2.044a1 1 0 01.205-1.251l1.267-1.114a7.05 7.05 0 010-2.227L1.821 7.773a1 1 0 01-.206-1.25l1.18-2.045a1 1 0 011.187-.447l1.598.54A6.993 6.993 0 017.51 3.456l.33-1.652zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
            </svg>
            <h3 className="text-sm font-medium text-gray-700">{t('generation.offloadingVramOptimization')}</h3>
          </div>

          {/* Enable Checkbox */}
          <label className="flex items-center gap-2 mb-4 cursor-pointer">
            <input
              type="checkbox"
              checked={offloadingEnabled}
              onChange={(e) => setOffloadingEnabled(e.target.checked)}
              className="w-4 h-4 text-blue-500 rounded focus:ring-2 focus:ring-blue-500"
            />
            <span className="text-sm font-medium text-gray-700">{t('generation.enableOffloading')}</span>
          </label>

          {offloadingEnabled && (
            <>
              {/* Mode Selection (Preset vs Manual) */}
              <div className="mb-4 space-y-2">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    value="preset"
                    checked={offloadingMode === 'preset'}
                    onChange={(e) => setOffloadingMode(e.target.value as OffloadingMode)}
                    className="w-4 h-4 text-blue-500 focus:ring-2 focus:ring-blue-500"
                  />
                  <span className="text-sm font-medium text-gray-700">{t('generation.presetRecommended')}</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    value="manual"
                    checked={offloadingMode === 'manual'}
                    onChange={(e) => setOffloadingMode(e.target.value as OffloadingMode)}
                    className="w-4 h-4 text-blue-500 focus:ring-2 focus:ring-blue-500"
                  />
                  <span className="text-sm font-medium text-gray-700">{t('generation.manualAdvanced')}</span>
                </label>
              </div>

              {/* Preset Dropdown */}
              {offloadingMode === 'preset' && (
                <div>
                  <label htmlFor="offloadingPreset" className="block text-sm font-medium text-gray-700 mb-1">
                    {t('generation.presetConfiguration')}
                  </label>
                  <select
                    id="offloadingPreset"
                    value={offloadingPreset}
                    onChange={(e) => setOffloadingPreset(e.target.value as OffloadingPreset)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="balanced">{t('generation.balanced')}</option>
                    <option value="aggressive">{t('generation.aggressive')}</option>
                    <option value="extreme">{t('generation.extreme')}</option>
                  </select>

                  {/* Info card for selected preset */}
                  <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
                    <div className="text-sm space-y-1">
                      <div className="font-medium text-blue-900">
                        {t(`generation.${offloadingPreset}Configuration`)}
                      </div>
                      <div className="text-blue-800">
                        <strong>{t('generation.gpuLayers')}:</strong> {PRESET_INFO[offloadingPreset].gpu_layers} / 28
                      </div>
                      <div className="text-blue-800">
                        <strong>{t('generation.vramSavings')}:</strong> {PRESET_INFO[offloadingPreset].vram_savings}
                      </div>
                      <div className="text-blue-800">
                        <strong>{t('generation.speed')}:</strong> {PRESET_INFO[offloadingPreset].slowdown} {t('generation.slowerThanNoOffloading')}
                      </div>
                      <div className="text-blue-700 text-xs mt-2">
                        {t(`generation.recommendedFor${offloadingPreset === 'balanced' ? '12GbGpus' : offloadingPreset === 'aggressive' ? '8to12GbGpus' : '6to8GbGpus'}`)}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Manual Slider */}
              {offloadingMode === 'manual' && (
                <div>
                  <label htmlFor="manualGpuLayers" className="block text-sm font-medium text-gray-700 mb-1">
                    {t('generation.gpuLayers')}: {manualGpuLayers}
                  </label>
                  <input
                    type="range"
                    id="manualGpuLayers"
                    min="1"
                    max="28"
                    value={manualGpuLayers}
                    onChange={(e) => setManualGpuLayers(parseInt(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>1 ({t('generation.moreVramSavings')})</span>
                    <span>28 ({t('generation.lessVramSavings')})</span>
                  </div>
                  <p className="text-xs text-gray-600 mt-2">
                    {t('generation.fewerGpuLayersInfo')}
                  </p>
                </div>
              )}
            </>
          )}
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
          {loading ? t('generation.startingGeneration') : t('generation.startGeneration')}
        </button>
      </form>
    </div>
  );
}
