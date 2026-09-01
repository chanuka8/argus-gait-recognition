/**
 * Layout persistence and validation utility for ARGUS AI UI.
 * Stores layout preferences (e.g. panel widths, split ratios) in localStorage
 * under the namespace 'argus_ui_layout'.
 * Validates all values against strictly enforced bounds.
 */

const STORAGE_KEY = 'argus_ui_layout';

export const LAYOUT_DEFAULTS = {
    dashboardDockWidth: 420,
    caseDetailsPanelWidth: 300,
    adminSplitRatio: 60, // percentage for quick-access panel
};

export const LAYOUT_BOUNDS = {
    dashboardDockWidth: { min: 300, max: 640 },
    caseDetailsPanelWidth: { min: 240, max: 480 },
    adminSplitRatio: { min: 35, max: 75 },
};

/**
 * Validate a layout value against its predefined bounds.
 * Returns the validated value, or the default if invalid.
 */
const validateValue = (key, value) => {
    const bounds = LAYOUT_BOUNDS[key];
    const defaultValue = LAYOUT_DEFAULTS[key];
    if (!bounds || typeof defaultValue === 'undefined') return value;

    if (typeof value !== 'number' || isNaN(value) || !isFinite(value)) {
        return defaultValue;
    }

    if (value < bounds.min) return bounds.min;
    if (value > bounds.max) return bounds.max;
    return value;
};

/**
 * Load the persisted layout configuration.
 * Gracefully falls back to defaults if storage is missing, inaccessible, or corrupted.
 */
export const loadLayoutPreferences = () => {
    try {
        if (typeof window === 'undefined' || !window.localStorage) {
            return { ...LAYOUT_DEFAULTS };
        }

        const raw = window.localStorage.getItem(STORAGE_KEY);
        if (!raw) {
            return { ...LAYOUT_DEFAULTS };
        }

        const parsed = JSON.parse(raw);
        if (!parsed || typeof parsed !== 'object') {
            return { ...LAYOUT_DEFAULTS };
        }

        return {
            dashboardDockWidth: validateValue('dashboardDockWidth', parsed.dashboardDockWidth),
            caseDetailsPanelWidth: validateValue('caseDetailsPanelWidth', parsed.caseDetailsPanelWidth),
            adminSplitRatio: validateValue('adminSplitRatio', parsed.adminSplitRatio),
        };
    } catch (err) {
        console.warn('ARGUS UI: Failed to read layout preferences, using defaults:', err);
        return { ...LAYOUT_DEFAULTS };
    }
};

/**
 * Save a specific layout preference with validation.
 */
export const saveLayoutPreference = (key, value) => {
    try {
        if (typeof window === 'undefined' || !window.localStorage) return;

        const validated = validateValue(key, value);
        const current = loadLayoutPreferences();
        const updated = {
            ...current,
            [key]: validated,
        };

        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(updated));
    } catch (err) {
        console.warn('ARGUS UI: Failed to persist layout preference:', err);
    }
};
