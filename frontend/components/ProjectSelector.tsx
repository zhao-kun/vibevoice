"use client";

import { useState } from "react";
import { useProject } from "@/lib/ProjectContext";
import { useRouter } from "next/navigation";

export default function ProjectSelector() {
  const { projects, selectProject, createProject, deleteProject } = useProject();
  const router = useRouter();
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [newProjectDescription, setNewProjectDescription] = useState("");

  const handleSelectProject = (projectId: string) => {
    selectProject(projectId);
    router.push("/voice-editor");
  };

  const handleCreateProject = async () => {
    if (newProjectName.trim()) {
      try {
        await createProject(newProjectName.trim(), newProjectDescription.trim());
        setNewProjectName("");
        setNewProjectDescription("");
        setShowCreateModal(false);
        // Will automatically navigate via the context update
        setTimeout(() => router.push("/voice-editor"), 100);
      } catch (err) {
        alert("Failed to create project. Please try again.");
      }
    }
  };

  const handleDeleteProject = async (projectId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm("Are you sure you want to delete this project?")) {
      try {
        await deleteProject(projectId);
      } catch (err) {
        alert("Failed to delete project. Please try again.");
      }
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex items-center justify-center p-8">
      <div className="max-w-6xl w-full">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-900 mb-4">VibeVoice</h1>
          <p className="text-xl text-gray-600">Speech Generation Studio</p>
          <p className="text-sm text-gray-500 mt-2">Select a project to get started</p>
        </div>

        {/* Projects Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          {projects.map((project) => (
            <div
              key={project.id}
              onClick={() => handleSelectProject(project.id)}
              className="bg-white rounded-xl shadow-md hover:shadow-xl transition-all duration-200 p-6 cursor-pointer border-2 border-transparent hover:border-blue-500 group"
            >
              <div className="flex items-start justify-between mb-4">
                <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center text-white font-bold text-xl">
                  {project.name.charAt(0).toUpperCase()}
                </div>
                {projects.length > 1 && (
                  <button
                    onClick={(e) => handleDeleteProject(project.id, e)}
                    className="opacity-0 group-hover:opacity-100 transition-opacity p-2 hover:bg-red-50 rounded-lg text-red-600"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                )}
              </div>

              <h3 className="text-xl font-semibold text-gray-900 mb-2">{project.name}</h3>
              <p className="text-sm text-gray-600 mb-4 line-clamp-2">
                {project.description || "No description"}
              </p>

              <div className="flex items-center text-xs text-gray-500 space-x-4">
                <div className="flex items-center space-x-1">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>Updated {new Date(project.updatedAt).toLocaleDateString()}</span>
                </div>
              </div>
            </div>
          ))}

          {/* Create New Project Card */}
          <div
            onClick={() => setShowCreateModal(true)}
            className="bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl shadow-md hover:shadow-xl transition-all duration-200 p-6 cursor-pointer flex flex-col items-center justify-center text-white min-h-[200px] group hover:scale-105"
          >
            <div className="w-16 h-16 bg-white/20 rounded-full flex items-center justify-center mb-4 group-hover:bg-white/30 transition-colors">
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold mb-2">Create New Project</h3>
            <p className="text-sm text-white/80">Start a new voice generation project</p>
          </div>
        </div>

        {/* Create Project Modal */}
        {showCreateModal && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-xl shadow-2xl max-w-md w-full p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">Create New Project</h2>

              <div className="space-y-4 mb-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Project Name *
                  </label>
                  <input
                    type="text"
                    value={newProjectName}
                    onChange={(e) => setNewProjectName(e.target.value)}
                    placeholder="Enter project name"
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    autoFocus
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Description
                  </label>
                  <textarea
                    value={newProjectDescription}
                    onChange={(e) => setNewProjectDescription(e.target.value)}
                    placeholder="Enter project description (optional)"
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
                    rows={3}
                  />
                </div>
              </div>

              <div className="flex space-x-3">
                <button
                  onClick={() => {
                    setShowCreateModal(false);
                    setNewProjectName("");
                    setNewProjectDescription("");
                  }}
                  className="flex-1 px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors font-medium"
                >
                  Cancel
                </button>
                <button
                  onClick={handleCreateProject}
                  disabled={!newProjectName.trim()}
                  className="flex-1 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Create Project
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
