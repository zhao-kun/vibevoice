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
