// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { formatNumber } from '../../features/modeling/fitting/session-format';
import { formatDate } from './job-presentation';

// Rendered values must not follow the browser language: an Italian browser
// would otherwise show 40.093 as "40,093" and 10069 as "10.069".

describe('UI number and date formatting', () => {
    const NativeNumberFormat = Intl.NumberFormat;
    const NativeDateTimeFormat = Intl.DateTimeFormat;

    beforeEach(() => {
        // Simulate a browser whose default locale is Italian.
        vi.spyOn(Intl, 'NumberFormat').mockImplementation(function (locale?: Intl.LocalesArgument, options?: Intl.NumberFormatOptions) {
            return new NativeNumberFormat(locale ?? 'it-IT', options);
        });
        vi.spyOn(Intl, 'DateTimeFormat').mockImplementation(function (locale?: Intl.LocalesArgument, options?: Intl.DateTimeFormatOptions) {
            return new NativeDateTimeFormat(locale ?? 'it-IT', options);
        });
    });

    afterEach(() => vi.restoreAllMocks());

    it('formats fitting values with a decimal point and comma grouping', () => {
        expect(formatNumber(40.093)).toBe('40.093');
        expect(formatNumber(-236.582)).toBe('-236.582');
        expect(formatNumber(10069)).toBe('10,069');
        expect(formatNumber(6.85317e-5)).toBe('6.85317E-5');
    });

    it('formats dates in English', () => {
        expect(formatDate('2026-11-19T09:51:00Z')).toMatch(/^Nov 19, 2026/);
    });
});
