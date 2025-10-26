export interface Speaker {
  id: string;
  name: string;
  voiceFile: string | null;
  content: string;
}

export interface SpeakerData {
  speakers: Speaker[];
  activeSpeakerId: string | null;
}

// Speaker Role Management Types
export interface SpeakerRole {
  id: string; // unique identifier for the role (for frontend UI)
  speakerId: string; // "Speaker 1", "Speaker 2", etc. (matches backend speaker_id)
  name: string; // speaker name (matches backend)
  description: string; // user description of the voice
  voiceFilename: string | null; // backend filename (null if no file uploaded)
  voiceFile: File | null; // local file being uploaded (frontend only)
  createdAt?: string; // ISO timestamp from backend
  updatedAt?: string; // ISO timestamp from backend
}

export interface VoiceFile {
  name: string; // original filename
  size: number; // file size in bytes
  type: string; // MIME type
  dataUrl: string; // base64 encoded audio data
  uploadedAt: string; // ISO timestamp
}

export type AudioFileExtension = '.wav' | '.mp3' | '.m4a' | '.flac';
export const ACCEPTED_AUDIO_TYPES = ['audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/x-m4a', 'audio/flac'];
export const ACCEPTED_AUDIO_EXTENSIONS: AudioFileExtension[] = ['.wav', '.mp3', '.m4a', '.flac'];
