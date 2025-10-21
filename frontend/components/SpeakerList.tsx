"use client";

import { Speaker } from "@/types/speaker";

interface SpeakerListProps {
  speakers: Speaker[];
  activeSpeakerId: string | null;
  onSelectSpeaker: (speakerId: string) => void;
  onAddSpeaker: () => void;
  onRemoveSpeaker: (speakerId: string) => void;
}

export default function SpeakerList({
  speakers,
  activeSpeakerId,
  onSelectSpeaker,
  onAddSpeaker,
  onRemoveSpeaker,
}: SpeakerListProps) {
  return (
    <div className="h-full flex flex-col bg-gray-50 border-r border-gray-200">
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-lg font-semibold text-gray-800">Speakers</h2>
      </div>

      <div className="flex-1 overflow-y-auto p-2">
        {speakers.map((speaker) => (
          <div
            key={speaker.id}
            className={`
              flex items-center justify-between p-3 mb-2 rounded-lg cursor-pointer
              transition-colors duration-150
              ${
                activeSpeakerId === speaker.id
                  ? "bg-blue-500 text-white"
                  : "bg-white text-gray-700 hover:bg-gray-100"
              }
            `}
            onClick={() => onSelectSpeaker(speaker.id)}
          >
            <div className="flex items-center space-x-2">
              <div className={`
                w-2 h-2 rounded-full
                ${speaker.voiceFile ? "bg-green-500" : "bg-gray-300"}
              `} />
              <span className="font-medium">{speaker.name}</span>
            </div>

            {speakers.length > 1 && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onRemoveSpeaker(speaker.id);
                }}
                className={`
                  p-1 rounded hover:bg-red-100
                  ${activeSpeakerId === speaker.id ? "text-white hover:text-red-600" : "text-gray-400 hover:text-red-600"}
                `}
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
        ))}
      </div>

      <div className="p-4 border-t border-gray-200">
        <button
          onClick={onAddSpeaker}
          className="w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors flex items-center justify-center space-x-2"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          <span>Add Speaker</span>
        </button>
      </div>
    </div>
  );
}
