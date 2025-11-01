'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useProject } from '@/lib/ProjectContext';
import { useLanguage } from '@/lib/i18n/LanguageContext';
import { SessionProvider } from '@/lib/SessionContext';
import { GenerationProvider, useGeneration } from '@/lib/GenerationContext';
import GenerationHistory from '@/components/GenerationHistory';
import CurrentGeneration from '@/components/CurrentGeneration';
import GenerationForm from '@/components/GenerationForm';

function GenerateVoiceContent() {
  const { currentProject } = useProject();
  const { currentGeneration } = useGeneration();
  const { t } = useLanguage();

  // Safety check - should not happen due to wrapper logic, but prevents errors
  if (!currentProject) {
    return null;
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center space-x-2 mb-1">
          <h1 className="text-2xl font-bold text-gray-900">{t('generation.pageTitle')}</h1>
          {currentProject && (
            <span className="px-3 py-1 bg-blue-100 text-blue-700 text-xs font-medium rounded-full">
              {currentProject.name}
            </span>
          )}
        </div>
        <p className="text-sm text-gray-500">
          {t('generation.pageSubtitle')}
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
              <h3 className="text-sm font-semibold text-blue-900 mb-2">{t('generation.howItWorks')}</h3>
              <ul className="text-sm text-blue-800 space-y-1">
                <li>{t('generation.step1')}</li>
                <li>{t('generation.step2')}</li>
                <li>{t('generation.step3')}</li>
                <li>{t('generation.step4')}</li>
                <li>{t('generation.step5')}</li>
              </ul>
            </div>

            {/* Technical Info */}
            <div className="bg-gray-100 border border-gray-300 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-gray-900 mb-2">{t('generation.modelInformation')}</h3>
              <div className="text-xs text-gray-700 space-y-1">
                <p><strong>float8_e4m3fn:</strong> {t('generation.float8Info')}</p>
                <p><strong>bf16:</strong> {t('generation.bf16Info')}</p>
                <p><strong>CFG Scale:</strong> {t('generation.cfgScaleInfo')}</p>
                <p><strong>Seed:</strong> {t('generation.seedInfo')}</p>
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
  const { t } = useLanguage();

  // Redirect to home page if no project is selected (after loading completes)
  useEffect(() => {
    if (!loading && !currentProject) {
      router.push('/');
    }
  }, [loading, currentProject, router]);

  // Show content when project is available
  const showContent = !loading && currentProject;

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
            <h1 className="text-2xl font-bold text-gray-900">{t('generation.pageTitle')}</h1>
            <p className="text-sm text-gray-500 mt-1">{t('generation.pageSubtitle')}</p>
          </header>

          <div className="flex-1 flex items-center justify-center bg-gray-50">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-gray-500">
                {loading ? t('generation.loadingProject') : t('generation.redirecting')}
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
