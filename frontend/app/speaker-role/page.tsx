"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useProject } from "@/lib/ProjectContext";
import { SpeakerRoleProvider } from "@/lib/SpeakerRoleContext";
import SpeakerRoleManager from "@/components/SpeakerRoleManager";

export default function SpeakerRolePage() {
  const router = useRouter();
  const { currentProject, loading } = useProject();
  const [mounted, setMounted] = useState(false);

  // Only render after client-side mount to avoid SSR/SSG mismatch
  useEffect(() => {
    setMounted(true);
  }, []);

  // Redirect to home page if no project is selected (after loading completes)
  useEffect(() => {
    if (mounted && !loading && !currentProject) {
      router.push('/');
    }
  }, [mounted, loading, currentProject, router]);

  // During SSR/SSG or before mount, show loading state
  if (!mounted || loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-500">Loading project...</p>
        </div>
      </div>
    );
  }

  // Show loading while redirecting
  if (!currentProject) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-500">Redirecting to project selection...</p>
        </div>
      </div>
    );
  }

  return (
    <SpeakerRoleProvider projectId={currentProject.id}>
      <div className="h-full flex flex-col">
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
      </div>
    </SpeakerRoleProvider>
  );
}
