/**
 * Backend API client for VibeVoice
 */

import type {
  CreateGenerationRequest,
  CreateGenerationResponse,
  CurrentGenerationResponse,
  ListGenerationsResponse,
  GetGenerationResponse
} from '@/types/generation';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || '/api/v1';

export interface Project {
  id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
}

export interface Speaker {
  speaker_id: string;
  name: string;
  description: string;
  voice_filename: string;
  created_at: string;
  updated_at: string;
}

export interface DialogSession {
  session_id: string;
  name: string;
  description: string;
  text_filename: string;
  created_at: string;
  updated_at: string;
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async fetch<T>(
    endpoint: string,
    options?: RequestInit
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({
          error: 'Unknown error',
          message: response.statusText
        }));
        throw new Error(error.message || error.error || response.statusText);
      }

      return await response.json();
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error('Network error');
    }
  }

  // ============ Projects API ============

  async listProjects(): Promise<{ projects: Project[]; count: number }> {
    return this.fetch('/projects');
  }

  async getProject(projectId: string): Promise<Project> {
    return this.fetch(`/projects/${encodeURIComponent(projectId)}`);
  }

  async createProject(data: {
    name: string;
    description?: string;
  }): Promise<Project> {
    return this.fetch('/projects', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateProject(
    projectId: string,
    data: { name?: string; description?: string }
  ): Promise<Project> {
    return this.fetch(`/projects/${encodeURIComponent(projectId)}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteProject(projectId: string): Promise<{ message: string; project_id: string }> {
    return this.fetch(`/projects/${encodeURIComponent(projectId)}`, {
      method: 'DELETE',
    });
  }

  // ============ Speakers API ============

  async listSpeakers(projectId: string): Promise<{ speakers: Speaker[]; count: number }> {
    return this.fetch(`/projects/${encodeURIComponent(projectId)}/speakers`);
  }

  async getSpeaker(projectId: string, speakerId: string): Promise<Speaker> {
    return this.fetch(
      `/projects/${encodeURIComponent(projectId)}/speakers/${encodeURIComponent(speakerId)}`
    );
  }

  async createSpeaker(
    projectId: string,
    data: {
      name: string;
      description?: string;
      voice_file: File;
    }
  ): Promise<Speaker> {
    const formData = new FormData();
    formData.append('name', data.name);
    if (data.description) {
      formData.append('description', data.description);
    }
    formData.append('voice_file', data.voice_file);

    const url = `${this.baseUrl}/projects/${encodeURIComponent(projectId)}/speakers`;
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        error: 'Unknown error',
        message: response.statusText
      }));
      throw new Error(error.message || error.error || response.statusText);
    }

    return await response.json();
  }

  async updateSpeaker(
    projectId: string,
    speakerId: string,
    data: { name?: string; description?: string }
  ): Promise<Speaker> {
    return this.fetch(
      `/projects/${encodeURIComponent(projectId)}/speakers/${encodeURIComponent(speakerId)}`,
      {
        method: 'PUT',
        body: JSON.stringify(data),
      }
    );
  }

  async deleteSpeaker(
    projectId: string,
    speakerId: string
  ): Promise<{ message: string; speaker_id: string }> {
    return this.fetch(
      `/projects/${encodeURIComponent(projectId)}/speakers/${encodeURIComponent(speakerId)}`,
      {
        method: 'DELETE',
      }
    );
  }

  getVoiceFileUrl(projectId: string, speakerId: string): string {
    return `${this.baseUrl}/projects/${encodeURIComponent(projectId)}/speakers/${encodeURIComponent(speakerId)}/voice`;
  }

  // ============ Dialog Sessions API ============

  async listSessions(projectId: string): Promise<{ sessions: DialogSession[]; count: number }> {
    return this.fetch(`/projects/${encodeURIComponent(projectId)}/sessions`);
  }

  async getSession(projectId: string, sessionId: string): Promise<DialogSession> {
    return this.fetch(
      `/projects/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(sessionId)}`
    );
  }

  async createSession(
    projectId: string,
    data: {
      name: string;
      description?: string;
      dialog_text: string;
    }
  ): Promise<DialogSession> {
    return this.fetch(`/projects/${encodeURIComponent(projectId)}/sessions`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateSession(
    projectId: string,
    sessionId: string,
    data: {
      name?: string;
      description?: string;
      dialog_text?: string;
    }
  ): Promise<DialogSession> {
    return this.fetch(
      `/projects/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(sessionId)}`,
      {
        method: 'PUT',
        body: JSON.stringify(data),
      }
    );
  }

  async deleteSession(
    projectId: string,
    sessionId: string
  ): Promise<{ message: string; session_id: string }> {
    return this.fetch(
      `/projects/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(sessionId)}`,
      {
        method: 'DELETE',
      }
    );
  }

  async getSessionText(
    projectId: string,
    sessionId: string
  ): Promise<{ session_id: string; dialog_text: string }> {
    return this.fetch(
      `/projects/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(sessionId)}/text`
    );
  }

  getSessionDownloadUrl(projectId: string, sessionId: string): string {
    return `${this.baseUrl}/projects/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(sessionId)}/download`;
  }

  // ============ Generations API ============

  async createGeneration(
    projectId: string,
    data: CreateGenerationRequest
  ): Promise<CreateGenerationResponse> {
    return this.fetch(`/projects/${encodeURIComponent(projectId)}/generations`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getCurrentGeneration(): Promise<CurrentGenerationResponse> {
    return this.fetch('/projects/generations/current');
  }

  async listGenerations(projectId: string): Promise<ListGenerationsResponse> {
    return this.fetch(`/projects/${encodeURIComponent(projectId)}/generations`);
  }

  async getGeneration(projectId: string, requestId: string): Promise<GetGenerationResponse> {
    return this.fetch(`/projects/${encodeURIComponent(projectId)}/generations/${encodeURIComponent(requestId)}`);
  }

  async deleteGeneration(
    projectId: string,
    requestId: string
  ): Promise<{ message: string; request_id: string }> {
    return this.fetch(
      `/projects/${encodeURIComponent(projectId)}/generations/${encodeURIComponent(requestId)}`,
      {
        method: 'DELETE',
      }
    );
  }

  async batchDeleteGenerations(
    projectId: string,
    requestIds: string[]
  ): Promise<{
    message: string;
    deleted_count: number;
    failed_count: number;
    deleted_ids: string[];
    failed_ids: string[];
  }> {
    return this.fetch(
      `/projects/${encodeURIComponent(projectId)}/generations/batch-delete`,
      {
        method: 'POST',
        body: JSON.stringify({ request_ids: requestIds }),
      }
    );
  }

  getGenerationDownloadUrl(projectId: string, requestId: string): string {
    return `${this.baseUrl}/projects/${encodeURIComponent(projectId)}/generations/${encodeURIComponent(requestId)}/download`;
  }
}

// Export singleton instance
export const api = new ApiClient();
