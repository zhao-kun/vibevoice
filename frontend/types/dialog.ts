export interface DialogLine {
  id: string;
  speakerId: string;
  content: string;
}

export interface SpeakerInfo {
  id: string;
  name: string;
  displayName: string;
}

export interface DialogSession {
  id: string;
  name: string;
  description: string;
  dialogLines: DialogLine[];
  createdAt: Date;
  updatedAt: Date;
}
