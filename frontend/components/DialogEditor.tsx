"use client";

import { useState, useEffect } from "react";
import { DialogLine, SpeakerInfo } from "@/types/dialog";
import toast from "react-hot-toast";

interface DialogEditorProps {
  dialogLines: DialogLine[];
  speakers: SpeakerInfo[];
  onUpdateLine: (lineId: string, content: string) => void;
  onDeleteLine: (lineId: string) => void;
  onMoveLine: (lineId: string, direction: "up" | "down") => void;
  onClear: () => void;
  onSave: () => void;
  onSetLines: (lines: DialogLine[]) => void;
  hasUnsavedChanges: boolean;
  isSaving: boolean;
}

export default function DialogEditor({
  dialogLines,
  speakers,
  onUpdateLine,
  onDeleteLine,
  onMoveLine,
  onClear,
  onSave,
  onSetLines,
  hasUnsavedChanges,
  isSaving,
}: DialogEditorProps) {
  const [viewMode, setViewMode] = useState<'visual' | 'text'>('visual');
  const [textContent, setTextContent] = useState('');

  const getSpeakerById = (speakerId: string) => {
    return speakers.find((s) => s.id === speakerId);
  };

  // Convert dialog lines to text format
  const dialogLinesToText = (lines: DialogLine[]): string => {
    return lines.map(line => {
      const speaker = getSpeakerById(line.speakerId);
      const speakerName = speaker?.displayName || line.speakerId;
      return `${speakerName}: ${line.content}`;
    }).join('\n\n');
  };

  // Parse text format to dialog lines
  const textToDialogLines = (text: string): DialogLine[] | null => {
    const lines = text.trim().split('\n\n');
    const parsed: DialogLine[] = [];

    for (const line of lines) {
      const trimmedLine = line.trim();
      if (!trimmedLine) continue;

      const match = trimmedLine.match(/^(.+?):\s*(.*)$/);
      if (!match) {
        toast.error(`Invalid format: "${trimmedLine.substring(0, 50)}..."`);
        return null;
      }

      const [, speakerName, content] = match;
      const speaker = speakers.find(s => s.displayName === speakerName.trim());

      if (!speaker) {
        toast.error(`Unknown speaker: "${speakerName}". Available speakers: ${speakers.map(s => s.displayName).join(', ')}`);
        return null;
      }

      parsed.push({
        id: `line-${Date.now()}-${Math.random()}`,
        speakerId: speaker.id,
        content: content.trim(),
      });
    }

    return parsed;
  };

  // Update text content when switching to text view
  useEffect(() => {
    if (viewMode === 'text') {
      setTextContent(dialogLinesToText(dialogLines));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [viewMode]);

  // Handle view mode toggle
  const handleToggleView = () => {
    if (viewMode === 'visual') {
      // Switch to text view
      setViewMode('text');
      setTextContent(dialogLinesToText(dialogLines));
    } else {
      // Switch back to visual view - validate and parse
      const parsed = textToDialogLines(textContent);
      if (parsed !== null) {
        // Valid! Update the dialog lines
        onSetLines(parsed);
        setViewMode('visual');
        toast.success('Text parsed successfully');
      }
      // If invalid, stay in text view (error already shown)
    }
  };

  return (
    <div className="h-full flex flex-col bg-white">
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-3">
            <h2 className="text-lg font-semibold text-gray-800">Dialog Editor</h2>
            {/* View Mode Toggle */}
            <div className="flex bg-gray-100 rounded-lg p-1">
              <button
                onClick={() => viewMode === 'text' && handleToggleView()}
                className={`px-3 py-1 text-sm font-medium rounded transition-colors ${
                  viewMode === 'visual'
                    ? 'bg-white text-blue-600 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                Visual
              </button>
              <button
                onClick={() => viewMode === 'visual' && handleToggleView()}
                className={`px-3 py-1 text-sm font-medium rounded transition-colors ${
                  viewMode === 'text'
                    ? 'bg-white text-blue-600 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                Text
              </button>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            {hasUnsavedChanges && (
              <span className="text-xs text-orange-600 bg-orange-50 px-3 py-1.5 rounded-lg">
                Unsaved changes
              </span>
            )}
            <button
              onClick={onClear}
              className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors font-medium"
            >
              Clear
            </button>
            <button
              onClick={onSave}
              disabled={!hasUnsavedChanges || isSaving}
              className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isSaving ? 'Saving...' : 'Save'}
            </button>
          </div>
        </div>
        <p className="text-sm text-gray-500">
          {dialogLines.length} dialog lines • {viewMode === 'visual' ? 'Edit content and manage sequence' : 'Edit as raw text (format: "Speaker Name: dialog text")'}
        </p>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {viewMode === 'text' ? (
          <textarea
            value={textContent}
            onChange={(e) => setTextContent(e.target.value)}
            placeholder="Enter dialog in the format:&#10;&#10;Speaker 1: Dialog text here&#10;&#10;Speaker 2: Another line"
            className="w-full h-full p-4 border border-gray-300 rounded-lg resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-700 font-mono text-sm"
          />
        ) : dialogLines.length === 0 ? (
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
          <div className="space-y-3">{
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
          })}
          </div>
        )}
      </div>
    </div>
  );
}
