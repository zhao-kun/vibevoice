"""
i18n utilities for backend API
"""
import json
from pathlib import Path
from flask import request, g
from functools import wraps

# Load translation files
_translations = {}
_i18n_dir = Path(__file__).parent

def load_translations():
    """Load all translation files"""
    global _translations
    for locale_file in _i18n_dir.glob('*.json'):
        locale = locale_file.stem
        with open(locale_file, 'r', encoding='utf-8') as f:
            _translations[locale] = json.load(f)

# Load translations on module import
load_translations()

SUPPORTED_LOCALES = list(_translations.keys())
DEFAULT_LOCALE = 'en'


def get_locale():
    """
    Get the best matching locale from request Accept-Language header
    or from custom X-Language header
    """
    # Check if we already determined the locale for this request
    if hasattr(g, 'locale'):
        return g.locale

    # Check custom header first (set by frontend)
    custom_lang = request.headers.get('X-Language')
    if custom_lang and custom_lang in SUPPORTED_LOCALES:
        g.locale = custom_lang
        return custom_lang

    # Parse Accept-Language header
    accept_language = request.headers.get('Accept-Language', '')

    # Simple parsing of Accept-Language
    # Format: "en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7"
    if accept_language:
        languages = []
        for lang_entry in accept_language.split(','):
            parts = lang_entry.strip().split(';')
            lang_code = parts[0].strip().lower()

            # Extract base language (e.g., 'zh' from 'zh-CN')
            base_lang = lang_code.split('-')[0]

            # Get quality factor (default 1.0)
            quality = 1.0
            if len(parts) > 1 and parts[1].startswith('q='):
                try:
                    quality = float(parts[1].split('=')[1])
                except (ValueError, IndexError):
                    quality = 1.0

            languages.append((base_lang, quality))

        # Sort by quality factor (descending)
        languages.sort(key=lambda x: x[1], reverse=True)

        # Find first supported locale
        for lang, _ in languages:
            if lang in SUPPORTED_LOCALES:
                g.locale = lang
                return lang

    # Default to English
    g.locale = DEFAULT_LOCALE
    return DEFAULT_LOCALE


def translate(key: str, **params) -> str:
    """
    Get translated message for the current locale

    Args:
        key: Translation key in dot notation (e.g., 'errors.not_found')
        **params: Parameters to format into the translation string

    Returns:
        Translated and formatted message
    """
    locale = get_locale()

    # Get translation from the locale
    keys = key.split('.')
    value = _translations.get(locale, _translations[DEFAULT_LOCALE])

    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            # Fallback to English
            value = _translations[DEFAULT_LOCALE]
            for fallback_key in keys:
                if isinstance(value, dict) and fallback_key in value:
                    value = value[fallback_key]
                else:
                    # Return key if not found
                    return key
            break

    # If value is not a string, return the key
    if not isinstance(value, str):
        return key

    # Format parameters
    if params:
        try:
            value = value.format(**params)
        except (KeyError, ValueError):
            pass  # Return unformatted if formatting fails

    return value


# Shorthand alias
t = translate


def with_locale(f):
    """
    Decorator to ensure locale is determined for the request
    Use this on routes that need i18n
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # This will set g.locale if not already set
        get_locale()
        return f(*args, **kwargs)
    return decorated_function
