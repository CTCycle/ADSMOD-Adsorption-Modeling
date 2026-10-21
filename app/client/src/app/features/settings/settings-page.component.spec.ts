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

describe('SettingsPageComponent confirmations', () => {
    beforeEach(async () => {
        fetchMock.mockReset().mockImplementation(async (input) =>
            String(input).includes('/database') ? databaseResponse() : apiResponse(),
        );
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

    it('asks in-page confirmation before resetting settings', async () => {
        const fixture = await createPage();
        const reset = fixture.nativeElement.querySelector('.settings-action-buttons .secondary') as HTMLButtonElement;
        reset.click();
        fixture.detectChanges();

        const dialog = fixture.nativeElement.querySelector('[role="alertdialog"]') as HTMLElement | null;
        expect(dialog?.textContent).toContain('Reset application settings?');
        (dialog?.querySelector('button') as HTMLButtonElement).click();
        fixture.detectChanges();
        await fixture.whenStable();

        expect(fixture.nativeElement.querySelector('[role="alertdialog"]')).toBeNull();
        expect(fetchMock).toHaveBeenCalledTimes(2);
    });

    it('keeps or discards an unsaved draft through the route guard dialog', async () => {
        const fixture = await createPage();
        const component = fixture.componentInstance;
        const port = fixture.nativeElement.querySelector('#setting-backend-port') as HTMLInputElement;
        port.value = '6046';
        port.dispatchEvent(new Event('input', { bubbles: true }));
        fixture.detectChanges();

        const keepPromise = component.canLeave() as Promise<boolean>;
        fixture.detectChanges();
        const keepDialog = fixture.nativeElement.querySelector('[role="alertdialog"]') as HTMLElement;
        expect(keepDialog.textContent).toContain('Discard unsaved changes?');
        (keepDialog.querySelector('button') as HTMLButtonElement).click();
        fixture.detectChanges();
        expect(await keepPromise).toBe(false);

        const discardPromise = component.canLeave() as Promise<boolean>;
        fixture.detectChanges();
        const discardDialog = fixture.nativeElement.querySelector('[role="alertdialog"]') as HTMLElement;
        const discardButton = Array.from(discardDialog.querySelectorAll('button')).find((button) => button.textContent?.includes('Discard changes'));
        discardButton?.click();
        fixture.detectChanges();
        expect(await discardPromise).toBe(true);
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
