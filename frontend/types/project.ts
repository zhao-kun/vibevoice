export interface Project {
  id: string;
  name: string;
  description: string;
  createdAt: Date;
  updatedAt: Date;
}

export interface ProjectContextType {
  currentProject: Project | null;
  projects: Project[];
  selectProject: (projectId: string) => void;
  createProject: (name: string, description: string) => Promise<void>;
  deleteProject: (projectId: string) => Promise<void>;
  updateProject: (projectId: string, updates: Partial<Project>) => Promise<void>;
  loading: boolean;
  error: string | null;
}
