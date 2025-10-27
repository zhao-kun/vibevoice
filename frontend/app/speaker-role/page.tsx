"use client";

import { useProject } from "@/lib/ProjectContext";
import { SpeakerRoleProvider } from "@/lib/SpeakerRoleContext";
import SpeakerRoleManager from "@/components/SpeakerRoleManager";

export default function SpeakerRolePage() {
  const { currentProject } = useProject();

  if (!currentProject) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-xl font-semibold text-gray-700 mb-2">No Project Selected</h2>
          <p className="text-gray-500">Please select a project to manage speaker roles</p>
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
