"use client";

import { Speaker } from "@/types/speaker";

interface TextEditorProps {
  speaker: Speaker | null;
  onContentChange: (content: string) => void;
}

export default function TextEditor({ speaker, onContentChange }: TextEditorProps) {
  if (!speaker) {
    return (
      <div className="h-full flex items-center justify-center bg-white">
        <div className="text-center text-gray-400">
          <svg className="w-16 h-16 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
          </svg>
          <p className="text-lg">Select a speaker to start editing</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-white">
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-lg font-semibold text-gray-800">
          Text Content - {speaker.name}
        </h2>
        <p className="text-sm text-gray-500 mt-1">
          Enter the text that {speaker.name} will speak
        </p>
      </div>

      <div className="flex-1 p-4">
        <textarea
          value={speaker.content}
          onChange={(e) => onContentChange(e.target.value)}
          placeholder={`Enter text for ${speaker.name}...`}
          className="w-full h-full p-4 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-700 leading-relaxed"
        />
      </div>

      <div className="p-4 border-t border-gray-200 bg-gray-50">
        <div className="flex items-center justify-between text-sm text-gray-600">
          <span>Character count: {speaker.content.length}</span>
          <span>Word count: {speaker.content.trim() ? speaker.content.trim().split(/\s+/).length : 0}</span>
        </div>
      </div>
    </div>
  );
}
