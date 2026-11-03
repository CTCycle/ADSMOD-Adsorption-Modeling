// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import type { UnitOption } from '../../models/dataset.model';

/** Mirrors the backend unit key: NFKC, case-folded, spaces removed. */
export function unitKey(value: string): string {
    return value
        .normalize('NFKC')
        .trim()
        .toLowerCase()
        .replace(/−/g, '-')
        .replace(/·/g, ' ')
        .replace(/³/g, '3')
        .replace(/\s+/g, '');
}

function unitKeys(option: UnitOption): string[] {
    return [option.symbol, option.label, ...option.aliases].map(unitKey);
}

export function findUnit(
    options: readonly UnitOption[],
    text: string,
): UnitOption | null {
    const key = unitKey(text);
    if (!key) {
        return null;
    }
    return options.find((option) => unitKeys(option).includes(key)) ?? null;
}

export function searchUnits(
    options: readonly UnitOption[],
    query: string,
): UnitOption[] {
    const key = unitKey(query);
    if (!key) {
        return [...options];
    }
    const lowered = query.trim().toLowerCase();
    return options.filter(
        (option) =>
            unitKeys(option).some((candidate) => candidate.includes(key)) ||
            option.name.toLowerCase().includes(lowered),
    );
}
