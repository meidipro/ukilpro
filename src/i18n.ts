// src/i18n.ts
import { translations } from './locales/translations';

type Lang = 'en' | 'bn';

interface I18nService {
    getLanguage: () => Lang;
    setLanguage: (lang: Lang) => void;
    t: (key: keyof typeof translations) => string;
    getAiLanguage: () => 'English' | 'Bengali';
}

let currentLang: Lang = (localStorage.getItem('lang') as Lang) || 'en';

const i18n: I18nService = {
    getLanguage: () => {
        return currentLang;
    },
    setLanguage: (lang: Lang) => {
        currentLang = lang;
        localStorage.setItem('lang', lang);
        window.dispatchEvent(new Event('languageChange'));
    },
    t: (key: keyof typeof translations) => {
        return translations[key]?.[currentLang] || key;
    },
    getAiLanguage: () => {
        return currentLang === 'bn' ? 'Bengali' : 'English';
    }
};

export { i18n };