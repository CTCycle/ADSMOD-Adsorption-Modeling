import { afterEach, describe, expect, it, vi } from 'vitest';
import { waitForBackendRestart } from './deep-learning.service';

describe('waitForBackendRestart', () => {
    afterEach(() => vi.unstubAllGlobals());

    it('resolves only after the backend went away and came back', async () => {
        const outcomes = [true, true, false, false, true, true];
        const fetchMock = vi.fn(async () => {
            const live = outcomes.shift() ?? true;
            if (!live) throw new TypeError('connection refused');
            return { ok: true } as Response;
        });
        vi.stubGlobal('fetch', fetchMock);

        await waitForBackendRestart(async () => undefined);

        expect(fetchMock).toHaveBeenCalledTimes(5);
        expect(outcomes).toHaveLength(1);
    });
});
