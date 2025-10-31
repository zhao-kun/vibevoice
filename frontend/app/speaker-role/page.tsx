"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useProject } from "@/lib/ProjectContext";
import { SpeakerRoleProvider } from "@/lib/SpeakerRoleContext";
import SpeakerRoleManager from "@/components/SpeakerRoleManager";

export default function SpeakerRolePage() {
  const router = useRouter();
  const { currentProject, loading } = useProject();

  // Redirect to home page if no project is selected (after loading completes)
  useEffect(() => {
    console.log('[SpeakerRole] State check:', { loading, hasProject: !!currentProject });
    if (!loading && !currentProject) {
      console.log('[SpeakerRole] Redirecting to home page');
      router.push('/');
    }
  }, [loading, currentProject, router]);

  // Show content when project is available
  const showContent = !loading && currentProject;

  console.log('[SpeakerRole] Render decision:', { loading, hasProject: !!currentProject, showContent });

  return (
    <div className="h-full flex flex-col">
      {showContent ? (
        <SpeakerRoleProvider projectId={currentProject.id}>
          <>
            {/* Header */}
            <header className="bg-white border-b border-gray-200 px-6 py-4">
              <div className="flex items-center space-x-2 mb-1">
                <h1 className="text-2xl font-bold text-gray-900">Create Speaker Role</h1>
                {currentProject && (
                  <span className="px-3 py-1 bg-blue-100 text-blue-700 text-xs font-medium rounded-full">
                    {currentProject.name}
                  </span>
                )}
              </div>
              <p className="text-sm text-gray-500">
                Manage speaker voices for your project
              </p>
            </header>

            {/* Content */}
            <div className="flex-1 overflow-hidden bg-gray-50">
              <SpeakerRoleManager />
            </div>
          </>
        </SpeakerRoleProvider>
      ) : (
        <>
          <header className="bg-white border-b border-gray-200 px-6 py-4">
            <h1 className="text-2xl font-bold text-gray-900">Create Speaker Role</h1>
            <p className="text-sm text-gray-500 mt-1">Manage speaker voices for your project</p>
          </header>

          <div className="flex-1 flex items-center justify-center bg-gray-50">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-gray-500">
                {loading ? 'Loading project...' : 'Redirecting to project selection...'}
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
