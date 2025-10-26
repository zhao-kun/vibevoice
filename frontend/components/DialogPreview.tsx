"use client";

import { DialogLine, SpeakerInfo } from "@/types/dialog";
import { useRef } from "react";

interface DialogPreviewProps {
  dialogLines: DialogLine[];
  speakers: SpeakerInfo[];
}

export default function DialogPreview({ dialogLines, speakers }: DialogPreviewProps) {
  const previewRef = useRef<HTMLPreElement>(null);

  const getSpeakerById = (speakerId: string) => {
    return speakers.find((s) => s.id === speakerId);
  };

  const generatePreviewText = () => {
    return dialogLines
      .map((line) => {
        const speaker = getSpeakerById(line.speakerId);
        return `${speaker?.name}: ${line.content}`;
      })
      .join("\n\n");
  };

  const handleCopyToClipboard = () => {
    const text = generatePreviewText();
    navigator.clipboard.writeText(text);
  };

  const handleDownload = () => {
    const text = generatePreviewText();
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "dialog_content.txt";
    a.click();
    URL.revokeObjectURL(url);
  };

  const previewText = generatePreviewText();

  return (
    <div className="h-full flex flex-col bg-gray-50 border-l border-gray-200">
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-lg font-semibold text-gray-800">Preview</h2>
        <p className="text-sm text-gray-500 mt-1">
          Output format for VibeVoice
        </p>
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {dialogLines.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-gray-400">
              <svg className="w-16 h-16 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              <p className="text-lg">No content to preview</p>
            </div>
          </div>
        ) : (
          <div className="bg-white rounded-lg border border-gray-300 p-4 shadow-sm">
            <pre
              ref={previewRef}
              className="font-mono text-sm text-gray-700 whitespace-pre-wrap break-words"
            >
              {previewText}
            </pre>
          </div>
        )}
      </div>

      {dialogLines.length > 0 && (
        <div className="p-4 border-t border-gray-200 bg-white space-y-2">
          <button
            onClick={handleCopyToClipboard}
            className="w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors flex items-center justify-center space-x-2"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
            <span>Copy to Clipboard</span>
          </button>

          <button
            onClick={handleDownload}
            className="w-full px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors flex items-center justify-center space-x-2"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            <span>Download as TXT</span>
          </button>

          <div className="text-xs text-gray-500 pt-2">
            <p className="font-medium mb-1">Format:</p>
            <p>• Speaker Name: Dialog content</p>
            <p>• Empty line between dialogs</p>
            <p>• Ready for VibeVoice processing</p>
          </div>
        </div>
      )}
    </div>
  );
}
