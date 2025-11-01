"use client";

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import type { Locale } from './config';
import { i18n } from './config';
import enMessages from './locales/en.json';
import zhMessages from './locales/zh.json';

type Messages = typeof enMessages;

interface LanguageContextType {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: (key: string, params?: Record<string, string | number>) => string;
  messages: Messages;
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

const messages: Record<Locale, Messages> = {
  en: enMessages,
  zh: zhMessages,
};

function getBrowserLocale(): Locale {
  if (typeof window === 'undefined') return i18n.defaultLocale;

  const browserLang = navigator.language.toLowerCase();

  // Check for exact match or language prefix
  if (browserLang.startsWith('zh')) {
    return 'zh';
  }

  return 'en';
}

function getSavedLocale(): Locale | null {
  if (typeof window === 'undefined') return null;

  const saved = localStorage.getItem('vibevoice-locale');
  if (saved && (saved === 'en' || saved === 'zh')) {
    return saved as Locale;
  }

  return null;
}

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(i18n.defaultLocale);
  const [isHydrated, setIsHydrated] = useState(false);

  useEffect(() => {
    // Mark as hydrated first
    setIsHydrated(true);

    // On mount, check for saved locale or use browser locale
    const savedLocale = getSavedLocale();
    const initialLocale = savedLocale || getBrowserLocale();
    setLocaleState(initialLocale);
  }, []);

  const setLocale = useCallback((newLocale: Locale) => {
    setLocaleState(newLocale);
    localStorage.setItem('vibevoice-locale', newLocale);
  }, []);

  const t = useCallback((key: string, params?: Record<string, string | number>): string => {
    const keys = key.split('.');
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let value: any = messages[locale];

    for (const k of keys) {
      if (value && typeof value === 'object' && k in value) {
        value = value[k];
      } else {
        // Fallback to English if key not found
        value = messages.en;
        for (const fallbackKey of keys) {
          if (value && typeof value === 'object' && fallbackKey in value) {
            value = value[fallbackKey];
          } else {
            return key; // Return key if not found even in English
          }
        }
        break;
      }
    }

    if (typeof value !== 'string') {
      return key;
    }

    // Replace parameters in the translation
    if (params) {
      return value.replace(/\{(\w+)\}/g, (match, paramKey) => {
        return paramKey in params ? String(params[paramKey]) : match;
      });
    }

    return value;
  }, [locale]);

  // Provide default context during SSR and initial render
  const value = {
    locale: isHydrated ? locale : i18n.defaultLocale,
    setLocale,
    t,
    messages: messages[isHydrated ? locale : i18n.defaultLocale]
  };

  return (
    <LanguageContext.Provider value={value}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  const context = useContext(LanguageContext);
  if (context === undefined) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
}
