'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useProject } from '@/lib/ProjectContext';
import { SessionProvider } from '@/lib/SessionContext';
import { GenerationProvider, useGeneration } from '@/lib/GenerationContext';
import GenerationHistory from '@/components/GenerationHistory';
import CurrentGeneration from '@/components/CurrentGeneration';
import GenerationForm from '@/components/GenerationForm';

function GenerateVoiceContent() {
  const { currentProject } = useProject();
  const { currentGeneration } = useGeneration();

  // Safety check - should not happen due to wrapper logic, but prevents errors
  if (!currentProject) {
    return null;
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center space-x-2 mb-1">
          <h1 className="text-2xl font-bold text-gray-900">Generate Voice</h1>
          {currentProject && (
            <span className="px-3 py-1 bg-blue-100 text-blue-700 text-xs font-medium rounded-full">
              {currentProject.name}
            </span>
          )}
        </div>
        <p className="text-sm text-gray-500">
          Generate speech from your scripts and voice samples
        </p>
      </header>

      {/* Main Content - Two Column Layout */}
      <div className="flex-1 overflow-hidden bg-gray-50">
        <div className="h-full grid grid-cols-2 gap-6 p-6">
          {/* Left Column - Generation History */}
          <div className="bg-white rounded-lg shadow-sm p-6 overflow-hidden flex flex-col">
            <GenerationHistory />
          </div>

          {/* Right Column - Current Generation or Form */}
          <div className="flex flex-col gap-6 overflow-y-auto">
            {/* Current Generation Status (if active) */}
            {currentGeneration && <CurrentGeneration />}

            {/* Generation Form (if no active generation) */}
            {!currentGeneration && (
              <div className="bg-white rounded-lg shadow-sm p-6">
                <GenerationForm />
              </div>
            )}

            {/* Info Card */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-blue-900 mb-2">How it works</h3>
              <ul className="text-sm text-blue-800 space-y-1">
                <li>1. Select a dialog session from your project</li>
                <li>2. Configure generation parameters (model type, CFG scale, seed)</li>
                <li>3. Click &ldquo;Start Generation&rdquo; to begin processing</li>
                <li>4. Monitor progress in the current generation panel</li>
                <li>5. Download completed audio from the history list</li>
              </ul>
            </div>

            {/* Technical Info */}
            <div className="bg-gray-100 border border-gray-300 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-gray-900 mb-2">Model Information</h3>
              <div className="text-xs text-gray-700 space-y-1">
                <p><strong>float8_e4m3fn:</strong> Optimized 8-bit model, Load faster with less memory and slow inferencing</p>
                <p><strong>bf16:</strong> Full precision model, higher quality but slower loading and faster inferencing</p>
                <p><strong>CFG Scale:</strong> Controls adherence to input (1.0-3.0 recommended)</p>
                <p><strong>Seed:</strong> Random seed for reproducible results</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function GenerateVoicePage() {
  const router = useRouter();
  const { currentProject, loading } = useProject();
  const [mounted, setMounted] = useState(false);

  // Only render after client-side mount to avoid SSR/SSG mismatch
  useEffect(() => {
    console.log('[GenerateVoice] Component mounted on client');
    setMounted(true);
  }, []);

  // Redirect to home page if no project is selected (after loading completes)
  useEffect(() => {
    console.log('[GenerateVoice] State check:', { mounted, loading, hasProject: !!currentProject });
    if (mounted && !loading && !currentProject) {
      console.log('[GenerateVoice] Redirecting to home page');
      router.push('/');
    }
  }, [mounted, loading, currentProject, router]);

  // Always render consistent wrapper to avoid hydration mismatch
  const showContent = mounted && !loading && currentProject;

  return (
    <div className="h-full flex flex-col">
      {showContent ? (
        <SessionProvider>
          <GenerationProvider projectId={currentProject.id}>
            <GenerateVoiceContent />
          </GenerationProvider>
        </SessionProvider>
      ) : (
        <>
          <header className="bg-white border-b border-gray-200 px-6 py-4">
            <h1 className="text-2xl font-bold text-gray-900">Generate Voice</h1>
            <p className="text-sm text-gray-500 mt-1">Generate speech from your scripts and voice samples</p>
          </header>

          <div className="flex-1 flex items-center justify-center bg-gray-50">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-gray-500">
                {!mounted || loading ? 'Loading project...' : 'Redirecting to project selection...'}
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
