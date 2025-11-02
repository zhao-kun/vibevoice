export interface DialogLine {
  id: string;
  speakerId: string;
  content: string;
}

export interface SpeakerInfo {
  id: string;
  name: string;
  displayName: string;
  description?: string;
}

export interface DialogSession {
  id: string; // Frontend ID (for UI, same as session_id from backend)
  sessionId: string; // Backend session_id
  name: string;
  description: string;
  textFilename: string | null; // Backend text_filename (null if not saved yet)
  dialogLines: DialogLine[]; // Frontend-only, loaded from backend text file
  createdAt?: Date; // From backend
  updatedAt?: Date; // From backend
}
