import { TestBed } from '@angular/core/testing';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CoreWorkspaceStore } from '../../core/state/core-workspace.store';
import type { SettingsResponse, UserSettings } from '../../models/settings.model';
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

function apiResponse(current: UserSettings = defaults): Response {
    const body: SettingsResponse = {
        status: 'success',
        current: structuredClone(current),
        defaults: structuredClone(defaults),
        restart_required: false,
        restart_required_fields: [],
        pending_restart_fields: [],
    };
    return { ok: true, status: 200, json: async () => body } as Response;
}

describe('SettingsPageComponent confirmations', () => {
    beforeEach(async () => {
        fetchMock.mockReset().mockResolvedValue(apiResponse());
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
        expect(fetchMock).toHaveBeenCalledTimes(1);
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
});
