"use client";

import React, { useState, useRef, useEffect } from "react";
import { convertToWav } from "@/lib/audioUtils";

interface VoiceRecorderProps {
  onSave: (file: File) => void;
}

export default function VoiceRecorder({ onSave }: VoiceRecorderProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    // Cleanup on unmount
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  const requestMicrophonePermission = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      setHasPermission(true);
      return stream;
    } catch (error) {
      console.error("Microphone permission denied:", error);
      setHasPermission(false);
      return null;
    }
  };

  const startRecording = async () => {
    // Request permission if not already granted
    let stream = streamRef.current;
    if (!stream) {
      stream = await requestMicrophonePermission();
      if (!stream) return;
    }

    // Clear previous recording
    audioChunksRef.current = [];
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
      setAudioUrl(null);
    }

    // Create MediaRecorder
    const mediaRecorder = new MediaRecorder(stream, {
      mimeType: 'audio/webm;codecs=opus'
    });

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        audioChunksRef.current.push(event.data);
      }
    };

    mediaRecorder.onstop = () => {
      const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
      const url = URL.createObjectURL(audioBlob);
      setAudioUrl(url);
    };

    mediaRecorderRef.current = mediaRecorder;
    mediaRecorder.start();
    setIsRecording(true);
    setIsPaused(false);
    setRecordingTime(0);

    // Start timer
    timerRef.current = setInterval(() => {
      setRecordingTime(prev => prev + 1);
    }, 1000);
  };

  const pauseRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.pause();
      setIsPaused(true);
      if (timerRef.current) clearInterval(timerRef.current);
    }
  };

  const resumeRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "paused") {
      mediaRecorderRef.current.resume();
      setIsPaused(false);
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsPaused(false);
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
  };

  const playRecording = () => {
    if (audioRef.current && audioUrl) {
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  const pausePlayback = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      setIsPlaying(false);
    }
  };

  const handleAudioEnded = () => {
    setIsPlaying(false);
  };

  const handleSave = async () => {
    if (!audioUrl || isSaving) return;

    setIsSaving(true);
    try {
      // Fetch the blob from the URL
      const response = await fetch(audioUrl);
      const webmBlob = await response.blob();

      // Convert WebM to WAV format for editing support
      const wavBlob = await convertToWav(webmBlob);

      // Create File object with WAV extension
      const file = new File([wavBlob], `recording-${Date.now()}.wav`, {
        type: 'audio/wav'
      });

      onSave(file);

      // Reset recorder
      setAudioUrl(null);
      setRecordingTime(0);
    } catch (error) {
      console.error("Failed to convert audio to WAV:", error);
      alert("Failed to save recording. Please try again.");
    } finally {
      setIsSaving(false);
    }
  };

  const handleDiscard = () => {
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    setAudioUrl(null);
    setRecordingTime(0);
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  // Initial permission request UI
  if (hasPermission === null) {
    return (
      <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center bg-white">
        <div className="flex flex-col items-center gap-4">
          <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center">
            <svg className="w-8 h-8 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
            </svg>
          </div>
          <div>
            <p className="text-sm font-medium text-gray-900">Record Voice</p>
            <p className="text-xs text-gray-500 mt-1">Click to enable microphone access</p>
          </div>
          <button
            onClick={requestMicrophonePermission}
            className="px-4 py-2 bg-red-600 text-white text-sm font-medium rounded-lg hover:bg-red-700 transition-colors"
          >
            Enable Microphone
          </button>
        </div>
      </div>
    );
  }

  // Permission denied UI
  if (hasPermission === false) {
    return (
      <div className="border-2 border-red-300 rounded-lg p-8 text-center bg-red-50">
        <div className="flex flex-col items-center gap-4">
          <div className="w-16 h-16 bg-red-200 rounded-full flex items-center justify-center">
            <svg className="w-8 h-8 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </div>
          <div>
            <p className="text-sm font-medium text-red-900">Microphone Access Denied</p>
            <p className="text-xs text-red-700 mt-1">Please enable microphone permissions in your browser settings</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="border border-gray-300 rounded-lg p-6 bg-white">
      {audioUrl && <audio ref={audioRef} src={audioUrl} onEnded={handleAudioEnded} />}

      <div className="flex flex-col items-center gap-4">
        {/* Recording Status */}
        <div className="flex items-center gap-3">
          {isRecording && (
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${isPaused ? 'bg-yellow-500' : 'bg-red-600 animate-pulse'}`} />
              <span className="text-sm font-medium text-gray-700">
                {isPaused ? 'Paused' : 'Recording'}
              </span>
            </div>
          )}
          <span className="text-2xl font-mono font-bold text-gray-900">
            {formatTime(recordingTime)}
          </span>
        </div>

        {/* Recording Controls */}
        {!audioUrl && (
          <div className="flex items-center gap-3">
            {!isRecording ? (
              <button
                onClick={startRecording}
                className="flex items-center gap-2 px-6 py-3 bg-red-600 text-white rounded-full hover:bg-red-700 transition-colors font-medium"
              >
                <div className="w-4 h-4 bg-white rounded-full" />
                Start Recording
              </button>
            ) : (
              <>
                {isPaused ? (
                  <button
                    onClick={resumeRecording}
                    className="p-3 bg-green-600 text-white rounded-full hover:bg-green-700 transition-colors"
                    title="Resume"
                  >
                    <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M6 4l10 6-10 6V4z" />
                    </svg>
                  </button>
                ) : (
                  <button
                    onClick={pauseRecording}
                    className="p-3 bg-yellow-600 text-white rounded-full hover:bg-yellow-700 transition-colors"
                    title="Pause"
                  >
                    <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M5 4h3v12H5V4zm7 0h3v12h-3V4z" />
                    </svg>
                  </button>
                )}
                <button
                  onClick={stopRecording}
                  className="p-3 bg-gray-700 text-white rounded-full hover:bg-gray-800 transition-colors"
                  title="Stop"
                >
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                    <rect x="5" y="5" width="10" height="10" />
                  </svg>
                </button>
              </>
            )}
          </div>
        )}

        {/* Playback Controls */}
        {audioUrl && (
          <div className="w-full space-y-4">
            <div className="flex items-center justify-center gap-3">
              <button
                onClick={isPlaying ? pausePlayback : playRecording}
                className="p-3 bg-blue-600 text-white rounded-full hover:bg-blue-700 transition-colors"
              >
                {isPlaying ? (
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M5 4h3v12H5V4zm7 0h3v12h-3V4z" />
                  </svg>
                ) : (
                  <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M6 4l10 6-10 6V4z" />
                  </svg>
                )}
              </button>
              <span className="text-sm text-gray-600">Preview recording</span>
            </div>

            {/* Save/Discard Buttons */}
            <div className="flex gap-2">
              <button
                onClick={handleSave}
                disabled={isSaving}
                className="flex-1 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {isSaving ? (
                  <>
                    <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Converting to WAV...
                  </>
                ) : (
                  'Save Recording'
                )}
              </button>
              <button
                onClick={handleDiscard}
                disabled={isSaving}
                className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Discard
              </button>
            </div>
          </div>
        )}

        {/* Instructions */}
        {!isRecording && !audioUrl && (
          <p className="text-xs text-gray-500 text-center max-w-xs">
            Click &ldquo;Start Recording&rdquo; to record your voice. You can pause and resume during recording.
          </p>
        )}
      </div>
    </div>
  );
}
