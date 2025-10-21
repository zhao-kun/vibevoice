"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useProject } from "@/lib/ProjectContext";
import { useState, useEffect, useRef } from "react";

interface MenuItem {
  id: string;
  label: string;
  path: string;
  icon: React.ReactNode;
}

const menuItems: MenuItem[] = [
  {
    id: "voice-editor",
    label: "Voice Contents Editor",
    path: "/voice-editor",
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
      </svg>
    ),
  },
  {
    id: "speaker-role",
    label: "Create Speaker Role",
    path: "/speaker-role",
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
      </svg>
    ),
  },
  {
    id: "generate-voice",
    label: "Generate Voice",
    path: "/generate-voice",
    icon: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
      </svg>
    ),
  },
];

export default function Navigation() {
  const pathname = usePathname();
  const router = useRouter();
  const { currentProject, projects, selectProject } = useProject();
  const [showProjectMenu, setShowProjectMenu] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowProjectMenu(false);
      }
    };

    if (showProjectMenu) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [showProjectMenu]);

  const handleChangeProject = (projectId: string) => {
    selectProject(projectId);
    setShowProjectMenu(false);
  };

  const handleGoHome = () => {
    setShowProjectMenu(false);
    router.push("/");
  };

  return (
    <nav className="w-64 bg-gray-900 text-white flex flex-col h-screen fixed left-0 top-0 z-50">
      {/* Logo/Header */}
      <div className="p-6 border-b border-gray-800">
        <h1 className="text-xl font-bold text-white">VibeVoice</h1>
        <p className="text-xs text-gray-400 mt-1">Speech Generation Studio</p>
      </div>

      {/* Current Project Display */}
      <div className="px-4 py-3 border-b border-gray-800 bg-gray-800/50 relative z-50">
        <div className="text-xs text-gray-400 mb-1">Current Project</div>
        <div className="relative" ref={dropdownRef}>
          <button
            onClick={() => setShowProjectMenu(!showProjectMenu)}
            className="w-full flex items-center justify-between px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors text-left"
          >
            <div className="flex items-center space-x-2 flex-1 min-w-0">
              <div className="w-6 h-6 bg-blue-500 rounded flex items-center justify-center text-xs font-bold flex-shrink-0">
                {currentProject?.name.charAt(0).toUpperCase()}
              </div>
              <span className="text-sm font-medium text-white truncate">
                {currentProject?.name || "No project"}
              </span>
            </div>
            <svg className="w-4 h-4 text-gray-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {/* Project Dropdown */}
          {showProjectMenu && (
            <div className="absolute top-full left-0 right-0 mt-2 bg-gray-800 rounded-lg shadow-xl border border-gray-700 overflow-hidden z-[100]">
              <div className="max-h-64 overflow-y-auto">
                {projects.map((project) => (
                  <button
                    key={project.id}
                    onClick={() => handleChangeProject(project.id)}
                    className={`w-full flex items-center space-x-2 px-3 py-2 hover:bg-gray-700 transition-colors text-left ${
                      currentProject?.id === project.id ? "bg-gray-700" : ""
                    }`}
                  >
                    <div className="w-6 h-6 bg-blue-500 rounded flex items-center justify-center text-xs font-bold flex-shrink-0">
                      {project.name.charAt(0).toUpperCase()}
                    </div>
                    <span className="text-sm text-white truncate">{project.name}</span>
                    {currentProject?.id === project.id && (
                      <svg className="w-4 h-4 text-blue-400 ml-auto flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                    )}
                  </button>
                ))}
              </div>
              <div className="border-t border-gray-700">
                <button
                  onClick={handleGoHome}
                  className="w-full px-3 py-2 hover:bg-gray-700 transition-colors text-left flex items-center space-x-2 text-blue-400"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                  </svg>
                  <span className="text-sm font-medium">New Project</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Menu Items */}
      <div className="flex-1 py-4">
        {menuItems.map((item) => {
          const isActive = pathname === item.path;

          return (
            <Link
              key={item.id}
              href={item.path}
              className={`
                flex items-center space-x-3 px-6 py-3 transition-all duration-200
                relative
                ${
                  isActive
                    ? "bg-blue-600 text-white"
                    : "text-gray-300 hover:bg-gray-800 hover:text-white"
                }
              `}
            >
              {/* Active indicator */}
              {isActive && (
                <div className="absolute left-0 top-0 bottom-0 w-1 bg-blue-400" />
              )}

              <div className={isActive ? "text-white" : "text-gray-400"}>
                {item.icon}
              </div>
              <span className="font-medium text-sm">{item.label}</span>
            </Link>
          );
        })}
      </div>

      {/* Footer */}
      <div className="p-6 border-t border-gray-800">
        <div className="text-xs text-gray-500">
          <p>Version 1.0.0</p>
          <p className="mt-1">© 2025 VibeVoice</p>
        </div>
      </div>
    </nav>
  );
}
