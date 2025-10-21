"use client";

import { DialogLine, SpeakerInfo } from "@/types/dialog";

interface DialogEditorProps {
  dialogLines: DialogLine[];
  speakers: SpeakerInfo[];
  onUpdateLine: (lineId: string, content: string) => void;
  onDeleteLine: (lineId: string) => void;
  onMoveLine: (lineId: string, direction: "up" | "down") => void;
}

export default function DialogEditor({
  dialogLines,
  speakers,
  onUpdateLine,
  onDeleteLine,
  onMoveLine,
}: DialogEditorProps) {
  const getSpeakerById = (speakerId: string) => {
    return speakers.find((s) => s.id === speakerId);
  };

  return (
    <div className="h-full flex flex-col bg-white">
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-lg font-semibold text-gray-800">Dialog Editor</h2>
        <p className="text-sm text-gray-500 mt-1">
          {dialogLines.length} dialog lines • Edit content and manage sequence
        </p>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {dialogLines.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-gray-400">
              <svg className="w-16 h-16 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
              <p className="text-lg mb-2">No dialog lines yet</p>
              <p className="text-sm">Click a speaker on the left to add a new dialog line</p>
            </div>
          </div>
        ) : (
          dialogLines.map((line, index) => {
            const speaker = getSpeakerById(line.speakerId);
            const isFirst = index === 0;
            const isLast = index === dialogLines.length - 1;

            return (
              <div key={line.id} className="bg-gray-50 rounded-lg border border-gray-200 p-4">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <div className="w-8 h-8 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs font-bold">
                      {speaker?.id}
                    </div>
                    <div>
                      <div className="font-medium text-gray-800">{speaker?.displayName}</div>
                      <div className="text-xs text-gray-500">Line {index + 1}</div>
                    </div>
                  </div>

                  <div className="flex items-center space-x-1">
                    {/* Move up button */}
                    <button
                      onClick={() => onMoveLine(line.id, "up")}
                      disabled={isFirst}
                      className={`p-1 rounded ${
                        isFirst
                          ? "text-gray-300 cursor-not-allowed"
                          : "text-gray-600 hover:bg-gray-200"
                      }`}
                      title="Move up"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                      </svg>
                    </button>

                    {/* Move down button */}
                    <button
                      onClick={() => onMoveLine(line.id, "down")}
                      disabled={isLast}
                      className={`p-1 rounded ${
                        isLast
                          ? "text-gray-300 cursor-not-allowed"
                          : "text-gray-600 hover:bg-gray-200"
                      }`}
                      title="Move down"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </button>

                    {/* Delete button */}
                    <button
                      onClick={() => onDeleteLine(line.id)}
                      className="p-1 rounded text-red-600 hover:bg-red-50"
                      title="Delete line"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  </div>
                </div>

                <textarea
                  value={line.content}
                  onChange={(e) => onUpdateLine(line.id, e.target.value)}
                  placeholder={`Enter dialog for ${speaker?.displayName}...`}
                  className="w-full p-3 border border-gray-300 rounded-md resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-700"
                  rows={3}
                />
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
