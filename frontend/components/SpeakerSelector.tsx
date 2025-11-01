"use client";

import { SpeakerInfo } from "@/types/dialog";
import { useLanguage } from "@/lib/i18n/LanguageContext";

interface SpeakerSelectorProps {
  speakers: SpeakerInfo[];
  selectedSpeakerId: string | null;
  onSelectSpeaker: (speakerId: string) => void;
}

export default function SpeakerSelector({
  speakers,
  selectedSpeakerId,
  onSelectSpeaker,
}: SpeakerSelectorProps) {
  const { t } = useLanguage();

  return (
    <div className="h-full flex flex-col bg-gray-50 border-r border-gray-200">
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-lg font-semibold text-gray-800">{t('voiceEditor.speakers')}</h2>
        <p className="text-xs text-gray-500 mt-1">{t('voiceEditor.selectSpeakerHint')}</p>
      </div>

      <div className="flex-1 overflow-y-auto p-2">
        {speakers.map((speaker) => {
          const isSelected = selectedSpeakerId === speaker.id;

          return (
            <button
              key={speaker.id}
              onClick={() => onSelectSpeaker(speaker.id)}
              className={`
                w-full flex items-center space-x-3 p-4 mb-2 rounded-lg
                transition-all duration-150 text-left
                ${
                  isSelected
                    ? "bg-blue-500 text-white shadow-md"
                    : "bg-white text-gray-700 hover:bg-gray-100 hover:shadow-sm"
                }
              `}
            >
              <div className={`
                w-10 h-10 rounded-full flex items-center justify-center font-semibold text-sm
                ${isSelected ? "bg-blue-600" : "bg-blue-100 text-blue-600"}
              `}>
                {speaker.id.replace('Speaker ', '')}
              </div>
              <div className="flex-1">
                <div className="font-medium">{speaker.displayName}</div>
              </div>
            </button>
          );
        })}
      </div>

      <div className="p-4 border-t border-gray-200 bg-white">
        <div className="text-xs text-gray-500">
          <p className="font-medium mb-1">{t('voiceEditor.quickActions')}</p>
          <p>• {t('voiceEditor.clickSpeakerToAdd')}</p>
          <p>• {t('voiceEditor.editInCenter')}</p>
        </div>
      </div>
    </div>
  );
}
