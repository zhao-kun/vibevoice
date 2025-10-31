"use client";

import { useState, useEffect, useMemo } from "react";
import { useRouter } from "next/navigation";
import SpeakerSelector from "@/components/SpeakerSelector";
import DialogEditor from "@/components/DialogEditor";
import DialogPreview from "@/components/DialogPreview";
import SessionManager from "@/components/SessionManager";
import { DialogLine, SpeakerInfo } from "@/types/dialog";
import { useProject } from "@/lib/ProjectContext";
import { useSession } from "@/lib/SessionContext";
import { SpeakerRoleProvider, useSpeakerRole } from "@/lib/SpeakerRoleContext";
import toast from "react-hot-toast";

function VoiceEditorContent() {
  const { currentProject } = useProject();
  const { currentSession, updateSessionDialogs } = useSession();
  const { speakerRoles } = useSpeakerRole();
  const [dialogLines, setDialogLines] = useState<DialogLine[]>([]);
  const [selectedSpeakerId, setSelectedSpeakerId] = useState<string | null>(null);
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  // Convert speaker roles to SpeakerInfo format
  const speakers: SpeakerInfo[] = useMemo(() => {
    return speakerRoles.map(role => ({
      id: role.speakerId,
      name: role.speakerId,
      displayName: role.name || role.speakerId,
    }));
  }, [speakerRoles]);

  // Auto-select first speaker when speakers load
  useEffect(() => {
    if (speakers.length > 0 && !selectedSpeakerId) {
      setSelectedSpeakerId(speakers[0].id);
    }
  }, [speakers, selectedSpeakerId]);

  // Load dialog lines from current session (only when session ID changes)
  useEffect(() => {
    if (currentSession) {
      setDialogLines(currentSession.dialogLines);
      setHasUnsavedChanges(false);
    } else {
      setDialogLines([]);
      setHasUnsavedChanges(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentSession?.id]);

  // Track unsaved changes
  useEffect(() => {
    if (!currentSession) return;

    // Compare only content, not IDs (to avoid false positives when switching views)
    const currentContent = dialogLines.map(line => ({ speakerId: line.speakerId, content: line.content }));
    const savedContent = currentSession.dialogLines.map(line => ({ speakerId: line.speakerId, content: line.content }));

    const isDifferent = JSON.stringify(currentContent) !== JSON.stringify(savedContent);
    setHasUnsavedChanges(isDifferent);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dialogLines]);

  const handleSelectSpeaker = (speakerId: string) => {
    setSelectedSpeakerId(speakerId);
    // Add a new dialog line for this speaker
    const newLine: DialogLine = {
      id: `line-${Date.now()}`,
      speakerId,
      content: "",
    };
    setDialogLines([...dialogLines, newLine]);
  };

  const handleUpdateLine = (lineId: string, content: string) => {
    setDialogLines(
      dialogLines.map((line) =>
        line.id === lineId ? { ...line, content } : line
      )
    );
  };

  const handleDeleteLine = (lineId: string) => {
    setDialogLines(dialogLines.filter((line) => line.id !== lineId));
  };

  const handleMoveLine = (lineId: string, direction: "up" | "down") => {
    const index = dialogLines.findIndex((line) => line.id === lineId);
    if (index === -1) return;

    const newLines = [...dialogLines];
    if (direction === "up" && index > 0) {
      [newLines[index - 1], newLines[index]] = [newLines[index], newLines[index - 1]];
    } else if (direction === "down" && index < newLines.length - 1) {
      [newLines[index], newLines[index + 1]] = [newLines[index + 1], newLines[index]];
    }
    setDialogLines(newLines);
  };

  const handleSave = async () => {
    if (!currentSession || !hasUnsavedChanges) return;

    setIsSaving(true);
    try {
      await updateSessionDialogs(currentSession.id, dialogLines);
      setHasUnsavedChanges(false);
    } catch (error) {
      console.error('Failed to save dialog:', error);
      toast.error('Failed to save changes. Please try again.');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center space-x-2 mb-1">
          <h1 className="text-2xl font-bold text-gray-900">Voice Contents Editor</h1>
          {currentProject && (
            <span className="px-3 py-1 bg-blue-100 text-blue-700 text-xs font-medium rounded-full">
              {currentProject.name}
            </span>
          )}
          {currentSession && (
            <span className="px-3 py-1 bg-purple-100 text-purple-700 text-xs font-medium rounded-full">
              {currentSession.name}
            </span>
          )}
        </div>
        <p className="text-sm text-gray-500">
          Create and edit dialog sequences for multi-speaker text-to-speech
        </p>
      </header>

      {/* Four-panel layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Session Manager - Far Left */}
        <div className="w-64 flex-shrink-0">
          <SessionManager />
        </div>

        {/* Speaker selector - Left */}
        <div className="w-56 flex-shrink-0">
          <SpeakerSelector
            speakers={speakers}
            selectedSpeakerId={selectedSpeakerId}
            onSelectSpeaker={handleSelectSpeaker}
          />
        </div>

        {/* Dialog editor - Center */}
        <div className="flex-1">
          <DialogEditor
            dialogLines={dialogLines}
            speakers={speakers}
            onUpdateLine={handleUpdateLine}
            onDeleteLine={handleDeleteLine}
            onMoveLine={handleMoveLine}
            onClear={() => setDialogLines([])}
            onSave={handleSave}
            onSetLines={setDialogLines}
            hasUnsavedChanges={hasUnsavedChanges}
            isSaving={isSaving}
          />
        </div>

        {/* Preview - Right */}
        <div className="w-96 flex-shrink-0">
          <DialogPreview dialogLines={dialogLines} speakers={speakers} />
        </div>
      </div>
    </div>
  );
}

export default function VoiceEditorPage() {
  const router = useRouter();
  const { currentProject, loading } = useProject();
  const [mounted, setMounted] = useState(false);

  // Only render after client-side mount to avoid SSR/SSG mismatch
  useEffect(() => {
    setMounted(true);
  }, []);

  // Redirect to home page if no project is selected (after loading completes)
  useEffect(() => {
    if (mounted && !loading && !currentProject) {
      router.push('/');
    }
  }, [mounted, loading, currentProject, router]);

  // During SSR/SSG or before mount, show loading state
  if (!mounted || loading) {
    return (
      <div className="h-full flex items-center justify-center" suppressHydrationWarning>
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-500">Loading project...</p>
        </div>
      </div>
    );
  }

  // Show loading while redirecting
  if (!currentProject) {
    return (
      <div className="h-full flex items-center justify-center" suppressHydrationWarning>
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-500">Redirecting to project selection...</p>
        </div>
      </div>
    );
  }

  return (
    <SpeakerRoleProvider projectId={currentProject.id}>
      <VoiceEditorContent />
    </SpeakerRoleProvider>
  );
}
