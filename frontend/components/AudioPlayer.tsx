"use client";

import React, { useRef, useState, useEffect } from "react";
import WaveSurfer from "wavesurfer.js";
import RegionsPlugin from "wavesurfer.js/dist/plugins/regions.js";

interface AudioPlayerProps {
  voiceFileUrl: string; // URL to fetch the audio file from backend
  voiceFileName: string; // Original filename
  onChangeVoice: (file: File) => void;
  onTrimAudio?: (startTime: number, endTime: number) => void;
  onRemoveVoice?: () => void;
}

export default function AudioPlayer({ voiceFileUrl, voiceFileName, onChangeVoice, onTrimAudio, onRemoveVoice }: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const waveformRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const regionsPluginRef = useRef<RegionsPlugin | null>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [fileSize, setFileSize] = useState<number>(0);
  const [fileType, setFileType] = useState<string>("");
  const [editMode, setEditMode] = useState(false);
  const [selectedRegion, setSelectedRegion] = useState<{ start: number; end: number } | null>(null);
  const [isTrimming, setIsTrimming] = useState(false);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleLoadedMetadata = () => {
      setDuration(audio.duration);
    };

    const handleTimeUpdate = () => {
      setCurrentTime(audio.currentTime);
    };

    const handleEnded = () => {
      setIsPlaying(false);
      setCurrentTime(0);
    };

    audio.addEventListener("loadedmetadata", handleLoadedMetadata);
    audio.addEventListener("timeupdate", handleTimeUpdate);
    audio.addEventListener("ended", handleEnded);

    return () => {
      audio.removeEventListener("loadedmetadata", handleLoadedMetadata);
      audio.removeEventListener("timeupdate", handleTimeUpdate);
      audio.removeEventListener("ended", handleEnded);
    };
  }, [voiceFileUrl]);

  // Fetch file metadata
  useEffect(() => {
    const fetchMetadata = async () => {
      try {
        const response = await fetch(voiceFileUrl, { method: 'HEAD' });
        const size = response.headers.get('Content-Length');
        const type = response.headers.get('Content-Type');
        if (size) setFileSize(parseInt(size, 10));
        if (type) setFileType(type);
      } catch (error) {
        console.error("Failed to fetch file metadata:", error);
      }
    };

    fetchMetadata();
  }, [voiceFileUrl]);

  // Initialize WaveSurfer when entering edit mode
  useEffect(() => {
    if (!editMode || !waveformRef.current) return;

    // Create regions plugin
    const regionsPlugin = RegionsPlugin.create();
    regionsPluginRef.current = regionsPlugin;

    // Initialize WaveSurfer
    const wavesurfer = WaveSurfer.create({
      container: waveformRef.current,
      waveColor: '#93c5fd',
      progressColor: '#3b82f6',
      cursorColor: '#1d4ed8',
      barWidth: 2,
      barGap: 1,
      height: 100,
      normalize: true,
      plugins: [regionsPlugin],
    });

    wavesurferRef.current = wavesurfer;

    // Load audio
    wavesurfer.load(voiceFileUrl);

    // Handle region updates
    regionsPlugin.on('region-updated', (region) => {
      setSelectedRegion({ start: region.start, end: region.end });
    });

    // Enable region creation with drag
    regionsPlugin.enableDragSelection({
      color: 'rgba(59, 130, 246, 0.3)',
    });

    // Cleanup
    return () => {
      wavesurfer.destroy();
      wavesurferRef.current = null;
      regionsPluginRef.current = null;
    };
  }, [editMode, voiceFileUrl]);

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
    } else {
      audio.play();
    }
    setIsPlaying(!isPlaying);
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    if (!audio) return;

    const newTime = parseFloat(e.target.value);
    audio.currentTime = newTime;
    setCurrentTime(newTime);
  };

  const formatTime = (time: number): string => {
    if (isNaN(time)) return "0:00";
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, "0")}`;
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  const handleChangeVoiceClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      onChangeVoice(file);
    }
    // Reset input so the same file can be selected again
    e.target.value = '';
  };

  const toggleEditMode = () => {
    setEditMode(!editMode);
    setSelectedRegion(null);
    if (isPlaying) {
      audioRef.current?.pause();
      setIsPlaying(false);
    }
  };

  const handleTrimAudio = async () => {
    if (!selectedRegion || !onTrimAudio) return;

    setIsTrimming(true);
    try {
      onTrimAudio(selectedRegion.start, selectedRegion.end);
      setEditMode(false);
      setSelectedRegion(null);
    } catch (error) {
      console.error("Failed to trim audio:", error);
    } finally {
      setIsTrimming(false);
    }
  };

  const handleClearRegion = () => {
    if (regionsPluginRef.current) {
      regionsPluginRef.current.clearRegions();
      setSelectedRegion(null);
    }
  };

  const handlePlaySelection = () => {
    const wavesurfer = wavesurferRef.current;
    if (!wavesurfer || !selectedRegion) return;

    // Stop any current playback
    if (wavesurfer.isPlaying()) {
      wavesurfer.pause();
    }

    // Play the selected region
    wavesurfer.play(selectedRegion.start, selectedRegion.end);
  };

  return (
    <div className="border border-gray-300 rounded-lg p-4 bg-gray-50">
      <audio ref={audioRef} src={voiceFileUrl} preload="metadata" />
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/wav,audio/mp3,audio/m4a,audio/flac,audio/webm"
        onChange={handleFileSelect}
        className="hidden"
      />

      <div className="flex items-center justify-between mb-3">
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-gray-900 truncate">
            {voiceFileName}
          </p>
          <p className="text-xs text-gray-500">
            {fileSize > 0 && formatFileSize(fileSize)} {fileType && " · " + fileType.split("/")[1]?.toUpperCase()}
          </p>
        </div>
        <div className="flex gap-2">
          {onTrimAudio && (
            <button
              onClick={toggleEditMode}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium rounded-lg transition-colors border ${
                editMode
                  ? 'text-red-600 bg-red-50 hover:bg-red-100 border-red-200 hover:border-red-300'
                  : 'text-purple-600 bg-purple-50 hover:bg-purple-100 border-purple-200 hover:border-purple-300'
              }`}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.121 14.121L19 19m-7-7l7-7m-7 7l-2.879 2.879M12 12L9.121 9.121m0 5.758a3 3 0 10-4.243 4.243 3 3 0 004.243-4.243zm0-5.758a3 3 0 10-4.243-4.243 3 3 0 004.243 4.243z" />
              </svg>
              {editMode ? 'Cancel Edit' : 'Edit Voice'}
            </button>
          )}
          <button
            onClick={handleChangeVoiceClick}
            className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-blue-600 bg-blue-50 hover:bg-blue-100 rounded-lg transition-colors border border-blue-200 hover:border-blue-300"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
            </svg>
            Change Voice
          </button>
          {onRemoveVoice && (
            <button
              onClick={onRemoveVoice}
              className="flex items-center gap-1.5 px-3 py-1.5 text-sm font-medium text-red-600 bg-white hover:bg-red-50 rounded-lg transition-colors border border-red-300 hover:border-red-400"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
              Remove Voice
            </button>
          )}
        </div>
      </div>

      {editMode ? (
        <>
          {/* Waveform Editor */}
          <div className="mb-3">
            <div ref={waveformRef} className="bg-white rounded border border-gray-300"></div>
            <p className="text-xs text-gray-600 mt-2">
              Click and drag on the waveform to select the region you want to keep
            </p>
            {selectedRegion && (
              <div className="flex items-center justify-between mt-2 p-2 bg-blue-50 border border-blue-200 rounded">
                <p className="text-xs text-blue-800">
                  Selected: {formatTime(selectedRegion.start)} - {formatTime(selectedRegion.end)}
                  <span className="ml-2">({formatTime(selectedRegion.end - selectedRegion.start)})</span>
                </p>
                <div className="flex gap-2">
                  <button
                    onClick={handlePlaySelection}
                    className="flex items-center gap-1 px-2 py-1 text-xs text-white bg-blue-600 hover:bg-blue-700 rounded transition-colors"
                  >
                    <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                      <path d="M6 4l10 6-10 6V4z" />
                    </svg>
                    Play Selection
                  </button>
                  <button
                    onClick={handleClearRegion}
                    className="text-xs text-blue-600 hover:text-blue-800 underline"
                  >
                    Clear
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Save/Cancel Buttons */}
          <div className="flex gap-2">
            <button
              onClick={handleTrimAudio}
              disabled={!selectedRegion || isTrimming}
              className="flex-1 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium"
            >
              {isTrimming ? 'Saving...' : 'Save Trimmed Audio'}
            </button>
          </div>
        </>
      ) : (
        /* Normal Playback Mode */
        <div className="flex items-center gap-3">
          <button
            onClick={togglePlay}
            className="flex-shrink-0 w-10 h-10 flex items-center justify-center bg-blue-600 hover:bg-blue-700 text-white rounded-full"
          >
            {isPlaying ? (
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                <path d="M5 4h3v12H5V4zm7 0h3v12h-3V4z" />
              </svg>
            ) : (
              <svg className="w-5 h-5 ml-0.5" fill="currentColor" viewBox="0 0 20 20">
                <path d="M6 4l10 6-10 6V4z" />
              </svg>
            )}
          </button>

          <div className="flex-1 flex flex-col gap-1">
            <input
              type="range"
              min="0"
              max={duration || 0}
              value={currentTime}
              onChange={handleSeek}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              style={{
                background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${(currentTime / duration) * 100}%, #e5e7eb ${(currentTime / duration) * 100}%, #e5e7eb 100%)`
              }}
            />
            <div className="flex justify-between text-xs text-gray-500">
              <span>{formatTime(currentTime)}</span>
              <span>{formatTime(duration)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
