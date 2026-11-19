// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

/**
 * Locale for every number and date the UI renders. It matches Angular's default
 * LOCALE_ID (used by the number and date pipes) and the English UI text, so a value
 * reads the same on every page whatever the browser language: "1,000.5" never
 * becomes "1.000,5" on one page and stays "1,000.5" on another.
 */
export const APP_LOCALE = 'en-US';
