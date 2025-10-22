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
      <div className="h-full flex flex-col bg-gray-50">
        <SpeakerRoleManager />
      </div>
    </SpeakerRoleProvider>
  );
}
