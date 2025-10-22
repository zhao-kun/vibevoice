"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import { Project, ProjectContextType } from "@/types/project";
import { api } from "./api";

const ProjectContext = createContext<ProjectContextType | undefined>(undefined);

export function ProjectProvider({ children }: { children: React.ReactNode }) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [currentProject, setCurrentProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load projects from backend on mount
  useEffect(() => {
    loadProjects();
  }, []);

  const loadProjects = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await api.listProjects();
      const projectsWithDates = response.projects.map((p) => ({
        id: p.id,
        name: p.name,
        description: p.description,
        createdAt: new Date(p.created_at),
        updatedAt: new Date(p.updated_at),
      }));

      setProjects(projectsWithDates);

      // Restore current project from localStorage
      const savedCurrentProjectId = localStorage.getItem("vibevoice_current_project");
      if (savedCurrentProjectId) {
        const current = projectsWithDates.find((p: Project) => p.id === savedCurrentProjectId);
        if (current) {
          setCurrentProject(current);
        } else if (projectsWithDates.length > 0) {
          setCurrentProject(projectsWithDates[0]);
        }
      } else if (projectsWithDates.length > 0) {
        setCurrentProject(projectsWithDates[0]);
      }
    } catch (err) {
      console.error('Failed to load projects:', err);
      setError(err instanceof Error ? err.message : 'Failed to load projects');
    } finally {
      setLoading(false);
    }
  };

  // Save current project ID to localStorage
  useEffect(() => {
    if (currentProject) {
      localStorage.setItem("vibevoice_current_project", currentProject.id);
    }
  }, [currentProject]);

  const selectProject = (projectId: string) => {
    const project = projects.find((p) => p.id === projectId);
    if (project) {
      setCurrentProject(project);
    }
  };

  const createProject = async (name: string, description: string) => {
    try {
      const newProject = await api.createProject({ name, description });
      const projectWithDates = {
        id: newProject.id,
        name: newProject.name,
        description: newProject.description,
        createdAt: new Date(newProject.created_at),
        updatedAt: new Date(newProject.updated_at),
      };

      setProjects([...projects, projectWithDates]);
      setCurrentProject(projectWithDates);
    } catch (err) {
      console.error('Failed to create project:', err);
      throw err;
    }
  };

  const deleteProject = async (projectId: string) => {
    if (projects.length <= 1) {
      alert("Cannot delete the last project");
      return;
    }

    try {
      await api.deleteProject(projectId);

      const newProjects = projects.filter((p) => p.id !== projectId);
      setProjects(newProjects);

      if (currentProject?.id === projectId) {
        setCurrentProject(newProjects[0]);
      }
    } catch (err) {
      console.error('Failed to delete project:', err);
      throw err;
    }
  };

  const updateProject = async (projectId: string, updates: Partial<Project>) => {
    try {
      const updated = await api.updateProject(projectId, {
        name: updates.name,
        description: updates.description,
      });

      const projectWithDates = {
        id: updated.id,
        name: updated.name,
        description: updated.description,
        createdAt: new Date(updated.created_at),
        updatedAt: new Date(updated.updated_at),
      };

      const updatedProjects = projects.map((p) =>
        p.id === projectId ? projectWithDates : p
      );
      setProjects(updatedProjects);

      if (currentProject?.id === projectId) {
        setCurrentProject(projectWithDates);
      }
    } catch (err) {
      console.error('Failed to update project:', err);
      throw err;
    }
  };

  return (
    <ProjectContext.Provider
      value={{
        currentProject,
        projects,
        selectProject,
        createProject,
        deleteProject,
        updateProject,
        loading,
        error,
      }}
    >
      {children}
    </ProjectContext.Provider>
  );
}

export function useProject() {
  const context = useContext(ProjectContext);
  if (context === undefined) {
    throw new Error("useProject must be used within a ProjectProvider");
  }
  return context;
}
