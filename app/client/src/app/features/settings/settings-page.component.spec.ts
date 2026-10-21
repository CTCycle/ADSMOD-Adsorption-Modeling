import { TestBed } from '@angular/core/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CoreWorkspaceStore } from '../../core/state/core-workspace.store';
import type { DatabaseSettings, SettingsResponse, UserSettings } from '../../models/settings.model';
import { SettingsPageComponent } from './settings-page.component';

const defaults: UserSettings = {
    runtime: { host: '127.0.0.1', backend_port: 6045, frontend_port: 5173 },
    datasets: { allowed_extensions: ['.csv', '.xls', '.xlsx'], column_detection_cutoff: 0.6 },
    nist: { parallel_tasks: 40 },
    public_data: {
        request_timeout_seconds: 20,
        retry_attempts: 3,
        retry_backoff_max_seconds: 30,
        pubchem_parallel_requests: 2,
        cod_max_interactive_results: 250,
        health_cache_ttl_seconds: 900,
        sync_batch_size: 50,
        sync_max_records_per_run: 200,
        sync_max_concurrent_jobs: 1,
        sync_response_size_limit_bytes: 67_108_864,
        sync_resume_on_startup: false,
        auto_sync_enabled: false,
        auto_sync_categories: [],
        auto_sync_interval_hours: 24,
        auto_sync_fraction: 1,
    },
    fitting: {
        default_max_iterations: 1000,
        max_iterations_upper_bound: 1_000_000_000,
        default_parameter_initial: 1,
        default_parameter_min: 0,
        default_parameter_max: 100,
        preview_row_limit: 20,
        best_model_metric: 'aicc',
    },
    jobs: { polling_interval: 1 },
    training: {
        use_jit: false,
        jit_backend: 'inductor',
        use_mixed_precision: false,
        dataloader_workers: 0,
        persistent_workers: false,
    },
};

const fetchMock = vi.fn<typeof fetch>();

function apiResponse(current: UserSettings = defaults, pendingRestartFields: string[] = []): Response {
    const body: SettingsResponse = {
        status: 'success',
        current: structuredClone(current),
        defaults: structuredClone(defaults),
        restart_required: pendingRestartFields.length > 0,
        restart_required_fields: ['runtime.frontend_port'],
        pending_restart_fields: pendingRestartFields,
    };
    return { ok: true, status: 200, json: async () => body } as Response;
}

function databaseResponse(): Response {
    const body: DatabaseSettings = {
        status: 'success',
        backend: 'sqlite',
        connected: true,
        sqlite_path: 'data/database.db',
        engine: null,
        host: null,
        port: null,
        database_name: null,
        username: null,
        ssl: false,
        ssl_ca: null,
        connect_timeout: 30,
        password_configured: false,
    };
    return { ok: true, status: 200, json: async () => body } as Response;
}

describe('SettingsPageComponent automatic persistence', () => {
    let persistedSettings: UserSettings;

    beforeEach(async () => {
        persistedSettings = structuredClone(defaults);
        fetchMock.mockReset().mockImplementation(async (input, init) => {
            const url = String(input);
            if (url.endsWith('/database/switch')) {
                return { ok: false, status: 503, json: async () => ({ detail: 'Database switch failed.' }) } as Response;
            }
            if (url.includes('/database')) return databaseResponse();
            if (init?.method === 'PUT') {
                persistedSettings = JSON.parse(String(init.body)) as UserSettings;
                const restartFields = persistedSettings.runtime.backend_port === defaults.runtime.backend_port
                    ? []
                    : ['runtime.backend_port'];
                return apiResponse(persistedSettings, restartFields);
            }
            return apiResponse(persistedSettings);
        });
        vi.stubGlobal('fetch', fetchMock);
        await TestBed.configureTestingModule({
            imports: [SettingsPageComponent],
            providers: [
                {
                    provide: CoreWorkspaceStore,
                    useValue: { loadCatalog: vi.fn().mockResolvedValue(undefined) },
                },
            ],
        }).compileComponents();
    });

    afterEach(() => {
        vi.useRealTimers();
        vi.unstubAllGlobals();
        TestBed.resetTestingModule();
    });

    async function createPage() {
        const fixture = TestBed.createComponent(SettingsPageComponent);
        fixture.detectChanges();
        await fixture.whenStable();
        fixture.detectChanges();
        return fixture;
    }

    async function advanceAutosave(fixture: Awaited<ReturnType<typeof createPage>>, milliseconds = 400) {
        await vi.advanceTimersByTimeAsync(milliseconds);
        for (let attempt = 0; attempt < 12; attempt += 1) await Promise.resolve();
        fixture.detectChanges();
    }

    function settingsWrites() {
        return fetchMock.mock.calls.filter(([, init]) => init?.method === 'PUT');
    }

    it('debounces edits and saves automatically without save, apply, cancel, or discard controls', async () => {
        const fixture = await createPage();
        vi.useFakeTimers();
        const port = fixture.nativeElement.querySelector('#setting-backend-port') as HTMLInputElement;
        port.value = '6046';
        port.dispatchEvent(new Event('input', { bubbles: true }));
        fixture.detectChanges();

        expect(settingsWrites()).toHaveLength(0);
        await advanceAutosave(fixture, 399);
        expect(settingsWrites()).toHaveLength(0);
        await advanceAutosave(fixture, 1);
        expect(settingsWrites()).toHaveLength(1);
        expect(JSON.parse(String(settingsWrites()[0][1]?.body)).runtime.backend_port).toBe(6046);
        expect(fixture.nativeElement.textContent).toContain('runtime startup settings still need a restart');
        const ordinaryButtons = Array.from(
            fixture.nativeElement.querySelectorAll('.settings-actions button'),
        ) as HTMLButtonElement[];
        expect(ordinaryButtons.map((button) => button.textContent?.trim())).toEqual(['Reset to defaults']);
        expect(fixture.nativeElement.querySelector('[role="alertdialog"]')).toBeNull();
    });

    it('autosaves controls across all five application-settings tabs', async () => {
        const fixture = await createPage();
        vi.useFakeTimers();

        const edits: Array<{
            tab: string;
            selector: string;
            value: string;
            event: string;
            verify: (settings: UserSettings) => unknown;
            expected: unknown;
        }> = [
            {
                tab: 'runtime',
                selector: '#setting-host',
                value: 'localhost',
                event: 'change',
                verify: (settings) => settings.runtime.host,
                expected: 'localhost',
            },
            {
                tab: 'import',
                selector: '#setting-detection-cutoff',
                value: '0.75',
                event: 'input',
                verify: (settings) => settings.datasets.column_detection_cutoff,
                expected: 0.75,
            },
            {
                tab: 'public-data',
                selector: '#setting-timeout',
                value: '22',
                event: 'input',
                verify: (settings) => settings.public_data.request_timeout_seconds,
                expected: 22,
            },
            {
                tab: 'fitting',
                selector: '#setting-best-metric',
                value: 'rmse',
                event: 'change',
                verify: (settings) => settings.fitting.best_model_metric,
                expected: 'rmse',
            },
            {
                tab: 'training',
                selector: '#setting-jit-backend',
                value: 'eager',
                event: 'change',
                verify: (settings) => settings.training.jit_backend,
                expected: 'eager',
            },
        ];

        for (const edit of edits) {
            const tab = fixture.nativeElement.querySelector('#settings-tab-' + edit.tab) as HTMLButtonElement;
            tab.click();
            fixture.detectChanges();
            const input = fixture.nativeElement.querySelector(edit.selector) as HTMLInputElement | HTMLSelectElement;
            input.value = edit.value;
            input.dispatchEvent(new Event(edit.event, { bubbles: true }));
            fixture.detectChanges();
            const before = settingsWrites().length;
            await advanceAutosave(fixture);
            expect(settingsWrites()).toHaveLength(before + 1);
            const written = JSON.parse(String(settingsWrites().at(-1)?.[1]?.body)) as UserSettings;
            expect(edit.verify(written)).toBe(edit.expected);
        }
    });

    it('serializes rapid edits and writes the newest complete settings snapshot last', async () => {
        const deferred: { releaseFirst?: () => void; firstStarted?: () => void } = {};
        let writeCount = 0;
        const firstRequestStarted = new Promise<void>((resolve) => { deferred.firstStarted = resolve; });
        fetchMock.mockImplementation(async (input, init) => {
            const url = String(input);
            if (url.includes('/database')) return databaseResponse();
            if (init?.method === 'PUT') {
                writeCount += 1;
                const next = JSON.parse(String(init.body)) as UserSettings;
                if (writeCount === 1) {
                    deferred.firstStarted?.();
                    await new Promise<void>((resolve) => { deferred.releaseFirst = resolve; });
                }
                persistedSettings = next;
                return apiResponse(next);
            }
            return apiResponse(persistedSettings);
        });
        const fixture = await createPage();
        vi.useFakeTimers();

        const backendPort = fixture.nativeElement.querySelector('#setting-backend-port') as HTMLInputElement;
        backendPort.value = '6046';
        backendPort.dispatchEvent(new Event('input', { bubbles: true }));
        fixture.detectChanges();
        await advanceAutosave(fixture);
        await firstRequestStarted;

        const frontendPort = fixture.nativeElement.querySelector('#setting-frontend-port') as HTMLInputElement;
        frontendPort.value = '5174';
        frontendPort.dispatchEvent(new Event('input', { bubbles: true }));
        fixture.detectChanges();
        await advanceAutosave(fixture);

        deferred.releaseFirst?.();
        for (let attempt = 0; attempt < 24; attempt += 1) await Promise.resolve();
        fixture.detectChanges();
        expect(settingsWrites()).toHaveLength(2);
        const writes = settingsWrites().map(([, init]) => JSON.parse(String(init?.body)) as UserSettings);
        expect(writes[0].runtime.backend_port).toBe(6046);
        expect(writes[1].runtime.backend_port).toBe(6046);
        expect(writes[1].runtime.frontend_port).toBe(5174);
        expect(persistedSettings.runtime.frontend_port).toBe(5174);
    });

    it('keeps failed edits visible and retries them through autosave or route navigation', async () => {
        let failNextWrite = true;
        fetchMock.mockImplementation(async (input, init) => {
            const url = String(input);
            if (url.includes('/database')) return databaseResponse();
            if (init?.method === 'PUT') {
                if (failNextWrite) {
                    failNextWrite = false;
                    return { ok: false, status: 503, json: async () => ({ detail: 'Database unavailable.' }) } as Response;
                }
                persistedSettings = JSON.parse(String(init.body)) as UserSettings;
                return apiResponse(persistedSettings);
            }
            return apiResponse(persistedSettings);
        });
        const fixture = await createPage();
        vi.useFakeTimers();
        const port = fixture.nativeElement.querySelector('#setting-backend-port') as HTMLInputElement;
        port.value = '6046';
        port.dispatchEvent(new Event('input', { bubbles: true }));
        fixture.detectChanges();
        await advanceAutosave(fixture);

        expect(fixture.nativeElement.textContent).toContain('Changes could not be saved');
        expect(port.value).toBe('6046');
        expect(fixture.nativeElement.querySelector('[role="alertdialog"]')).toBeNull();

        expect(await fixture.componentInstance.canLeave()).toBe(true);
        expect(persistedSettings.runtime.backend_port).toBe(6046);
        expect(fixture.nativeElement.textContent).toContain('Saved and applied.');
    });

    it('keeps database edits unapplied until the explicit switch action', async () => {
        const fixture = await createPage();
        const databaseTab = fixture.nativeElement.querySelector('#settings-tab-database') as HTMLButtonElement;
        databaseTab.click();
        fixture.detectChanges();
        const path = fixture.nativeElement.querySelector('#database-sqlite-path') as HTMLInputElement;
        path.value = 'data/another.db';
        path.dispatchEvent(new Event('input', { bubbles: true }));
        fixture.detectChanges();

        expect(settingsWrites()).toHaveLength(0);
        expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith('/database/switch'))).toBe(false);

        const switchButton = Array.from(fixture.nativeElement.querySelectorAll('button') as NodeListOf<HTMLButtonElement>)
            .find((button) => button.textContent?.includes('Switch database')) as HTMLButtonElement;
        switchButton.click();
        fixture.detectChanges();
        await fixture.whenStable();
        expect(fetchMock.mock.calls.some(([url, init]) =>
            String(url).endsWith('/database/switch') && init?.method === 'POST',
        )).toBe(true);
        expect(settingsWrites()).toHaveLength(0);
    });

    it('autosaves reset-to-defaults without a confirmation dialog', async () => {
        const fixture = await createPage();
        vi.useFakeTimers();
        const port = fixture.nativeElement.querySelector('#setting-backend-port') as HTMLInputElement;
        port.value = '6046';
        port.dispatchEvent(new Event('input', { bubbles: true }));
        fixture.detectChanges();
        await advanceAutosave(fixture);
        expect(settingsWrites()).toHaveLength(1);
        const reset = Array.from(fixture.nativeElement.querySelectorAll('.settings-actions button') as NodeListOf<HTMLButtonElement>)
            .find((button) => button.textContent?.includes('Reset to defaults')) as HTMLButtonElement;
        reset.click();
        fixture.detectChanges();
        await advanceAutosave(fixture);

        expect(settingsWrites()).toHaveLength(2);
        expect(JSON.parse(String(settingsWrites()[1][1]?.body)).runtime.backend_port).toBe(defaults.runtime.backend_port);
        expect(fixture.nativeElement.querySelector('[role="alertdialog"]')).toBeNull();
    });

    it('keeps database settings available when application settings cannot load', async () => {
        fetchMock.mockImplementation(async (input) => {
            if (String(input).includes('/database')) return databaseResponse();
            return { ok: false, status: 503, json: async () => ({ detail: 'Settings unavailable.' }) } as Response;
        });
        const fixture = await createPage();
        const buttons = fixture.nativeElement.querySelectorAll('button') as NodeListOf<HTMLButtonElement>;
        const databaseButton = Array.from(buttons)
            .find((button) => button.textContent?.includes('Database settings')) as HTMLButtonElement;
        databaseButton.click();
        fixture.detectChanges();

        expect(fixture.nativeElement.textContent).toContain('Database');
        expect(fixture.nativeElement.querySelector('#database-backend')).not.toBeNull();
        expect(fixture.nativeElement.textContent).toContain('SQLite');
    });

    it('shows the restart marker only when the active database has pending startup settings', async () => {
        const current = await createPage();
        expect(current.nativeElement.querySelector('.restart-tag')).toBeNull();
        current.destroy();

        fetchMock.mockImplementation(async (input) =>
            String(input).includes('/database')
                ? databaseResponse()
                : apiResponse(defaults, ['runtime.frontend_port']),
        );
        const pending = await createPage();
        expect(pending.nativeElement.querySelector('.restart-tag')?.textContent).toContain('Restart required');
        expect(pending.nativeElement.querySelector('.settings-message.restart')).not.toBeNull();
    });
});
