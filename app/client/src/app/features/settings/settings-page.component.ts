import { CommonModule } from '@angular/common';
import { ChangeDetectorRef, Component, HostListener, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CoreWorkspaceStore } from '../../core/state/core-workspace.store';
import type { SettingsModelMetric, SettingsResponse, UserSettings } from '../../models/settings.model';
import { fetchSettings, resetSettings, saveSettings } from '../../services/settings.service';

type SettingsTab = 'runtime' | 'import' | 'public-data' | 'fitting' | 'training';
type SettingExtension = '.csv' | '.xls' | '.xlsx';
type SyncCategory = 'experiments' | 'guest' | 'host';

const TAB_ORDER: readonly SettingsTab[] = ['runtime', 'import', 'public-data', 'fitting', 'training'];

@Component({
    selector: 'adsmod-settings-page',
    standalone: true,
    imports: [CommonModule, FormsModule],
    template: `
        <section class="settings-page" aria-labelledby="settings-heading">
            @if (dialogMode; as mode) {
                <div class="settings-dialog-backdrop">
                    <section
                        class="settings-confirm-dialog"
                        role="alertdialog"
                        aria-modal="true"
                        aria-labelledby="settings-confirm-title"
                        aria-describedby="settings-confirm-description"
                        (keydown)="handleDialogKeydown($event)"
                    >
                        <p class="settings-eyebrow">Confirm action</p>
                        <h2 id="settings-confirm-title">{{ mode === 'reset' ? 'Reset application settings?' : 'Discard unsaved changes?' }}</h2>
                        <p id="settings-confirm-description">
                            {{ mode === 'reset' ? 'All application settings will be restored to their defaults and saved immediately.' : 'Your edits will be lost if you leave this page.' }}
                        </p>
                        <div class="settings-dialog-actions">
                            <button class="button secondary" type="button" (click)="dismissDialog()">{{ mode === 'reset' ? 'Cancel' : 'Keep editing' }}</button>
                            <button class="button primary" id="settings-confirm-primary" type="button" (click)="confirmDialog()">{{ mode === 'reset' ? 'Reset settings' : 'Discard changes' }}</button>
                        </div>
                    </section>
                </div>
            }
            <div class="settings-intro">
                <div>
                    <p class="settings-eyebrow">Workspace preferences</p>
                    <h2 id="settings-heading">Application settings</h2>
                    <p class="settings-lede">Tune runtime behavior, data acquisition, fitting and training defaults for future work.</p>
                </div>
                <div class="settings-intro-meta">
                    <span class="settings-scope-pill">SQLite-backed</span>
                    @if (dirty) {
                        <span class="settings-dirty-pill">Unsaved changes</span>
                    }
                </div>
            </div>

            @if (loading) {
                <div class="settings-state-card" role="status">Loading saved settings…</div>
            } @else if (loadError) {
                <div class="settings-message error" role="alert">
                    <span>{{ loadError }}</span>
                    <button class="button secondary" type="button" (click)="load()">Retry</button>
                </div>
            } @else if (draft; as settings) {
                @if (settings.pending_restart_fields.length) {
                    <div class="settings-message restart" role="status">
                        <span><strong>Restart required.</strong> {{ restartDescription(settings.pending_restart_fields) }}</span>
                    </div>
                }
                @if (validationErrors.length) {
                    <div class="settings-message error" role="alert">
                        <strong>Review these settings before saving:</strong>
                        <ul>@for (message of validationErrors; track message) { <li>{{ message }}</li> }</ul>
                    </div>
                }
                @if (errorMessage) {
                    <div class="settings-message error" role="alert">{{ errorMessage }}</div>
                }
                @if (statusMessage) {
                    <div class="settings-message success" role="status">{{ statusMessage }}</div>
                }

                <div class="settings-workspace">
                    <div class="settings-tabs" role="tablist" aria-label="Settings categories" (keydown)="handleTabKeydown($event)">
                        @for (tab of tabs; track tab.id) {
                            <button
                                class="settings-tab"
                                type="button"
                                role="tab"
                                [id]="'settings-tab-' + tab.id"
                                [attr.aria-selected]="activeTab === tab.id"
                                [attr.aria-controls]="'settings-panel'"
                                [attr.tabindex]="activeTab === tab.id ? 0 : -1"
                                (click)="activeTab = tab.id"
                            >
                                <span class="settings-tab-mark" aria-hidden="true">{{ tab.mark }}</span>
                                <span>{{ tab.label }}</span>
                            </button>
                        }
                    </div>

                    <div class="settings-panel" role="tabpanel" id="settings-panel" [attr.aria-labelledby]="'settings-tab-' + activeTab" tabindex="0">
                        @if (activeTab === 'runtime') {
                            <div class="settings-section-heading">
                                <div><p class="settings-eyebrow">Connection</p><h3>Runtime</h3><p>Host and port changes need a restart. Other saved preferences apply to new requests and jobs.</p></div>
                                <span class="restart-tag">Restart required</span>
                            </div>
                            <div class="settings-callout"><strong>Database bootstrap</strong><span>Database selection, credentials, storage root and the local settings database path remain in <code>settings/.env</code>. ADSMOD never returns database passwords.</span></div>
                            <div class="settings-grid">
                                <label class="settings-field" for="setting-host"><span class="settings-label">Backend host</span><select id="setting-host" class="select-input" [(ngModel)]="settings.runtime.host" (ngModelChange)="markChanged()"><option value="127.0.0.1">127.0.0.1</option><option value="localhost">localhost</option><option value="::1">::1</option></select><span class="settings-help">ADSMOD accepts loopback addresses only.</span></label>
                                <label class="settings-field" for="setting-backend-port"><span class="settings-label">Backend port</span><input id="setting-backend-port" type="number" min="1024" max="65535" step="1" [(ngModel)]="settings.runtime.backend_port" (ngModelChange)="markChanged()" /><span class="settings-help">API listener port · 1024–65535.</span></label>
                                <label class="settings-field" for="setting-frontend-port"><span class="settings-label">Frontend port</span><input id="setting-frontend-port" type="number" min="1024" max="65535" step="1" [(ngModel)]="settings.runtime.frontend_port" (ngModelChange)="markChanged()" /><span class="settings-help">Console listener port · must differ from the backend port.</span></label>
                            </div>
                        } @else if (activeTab === 'import') {
                            <div class="settings-section-heading"><div><p class="settings-eyebrow">Files and previews</p><h3>Data import</h3><p>Choose accepted spreadsheet formats and how confidently ADSMOD suggests column roles.</p></div></div>
                            <div class="settings-grid">
                                <fieldset class="settings-field settings-field-wide"><legend class="settings-label">Allowed file extensions</legend><span class="settings-help">Only formats supported by the import parser are available.</span><div class="settings-choice-row">@for (extension of extensions; track extension) { <label class="settings-choice"><input type="checkbox" [checked]="settings.datasets.allowed_extensions.includes(extension)" (change)="toggleExtension(settings, extension, $event)" /><span>{{ extension }}</span></label> }</div></fieldset>
                                <label class="settings-field" for="setting-detection-cutoff"><span class="settings-label">Detection confidence cutoff <strong>{{ percent(settings.datasets.column_detection_cutoff) }}</strong></span><input id="setting-detection-cutoff" type="range" min="0" max="1" step="0.01" [(ngModel)]="settings.datasets.column_detection_cutoff" (ngModelChange)="markChanged()" /><span class="settings-help">Column suggestions below this score are marked for manual mapping.</span></label>
                                <label class="settings-field" for="setting-preview-rows"><span class="settings-label">Preview row limit</span><input id="setting-preview-rows" type="number" min="1" max="500" step="1" [(ngModel)]="settings.fitting.preview_row_limit" (ngModelChange)="markChanged()" /><span class="settings-help">Maximum source rows shown in the import preview.</span></label>
                            </div>
                        } @else if (activeTab === 'public-data') {
                            <div class="settings-section-heading"><div><p class="settings-eyebrow">Providers and acquisition</p><h3>Public data</h3><p>Set request pacing, safe response limits and scheduled NIST refresh behavior.</p></div></div>
                            <div class="settings-subsection"><h4>Provider requests</h4><div class="settings-grid">
                                <label class="settings-field" for="setting-nist-concurrency"><span class="settings-label">NIST parallel requests</span><input id="setting-nist-concurrency" type="number" min="1" max="100" [(ngModel)]="settings.nist.parallel_tasks" (ngModelChange)="markChanged()" /><span class="settings-help">Concurrent NIST index and record requests.</span></label>
                                <label class="settings-field" for="setting-timeout"><span class="settings-label">Request timeout (seconds)</span><input id="setting-timeout" type="number" min="0.1" max="120" step="0.5" [(ngModel)]="settings.public_data.request_timeout_seconds" (ngModelChange)="markChanged()" /><span class="settings-help">Maximum time for each provider request.</span></label>
                                <label class="settings-field" for="setting-retries"><span class="settings-label">Retry attempts</span><input id="setting-retries" type="number" min="1" max="6" step="1" [(ngModel)]="settings.public_data.retry_attempts" (ngModelChange)="markChanged()" /><span class="settings-help">Attempts per provider request, including the first attempt.</span></label>
                                <label class="settings-field" for="setting-backoff"><span class="settings-label">Maximum retry backoff (seconds)</span><input id="setting-backoff" type="number" min="0.1" max="120" step="0.5" [(ngModel)]="settings.public_data.retry_backoff_max_seconds" (ngModelChange)="markChanged()" /><span class="settings-help">Upper bound for delay between retries.</span></label>
                                <label class="settings-field" for="setting-pubchem-concurrency"><span class="settings-label">PubChem parallel requests</span><input id="setting-pubchem-concurrency" type="number" min="1" max="3" [(ngModel)]="settings.public_data.pubchem_parallel_requests" (ngModelChange)="markChanged()" /><span class="settings-help">Bounded to PubChem’s supported request rate.</span></label>
                                <label class="settings-field" for="setting-cod-limit"><span class="settings-label">COD interactive result limit</span><input id="setting-cod-limit" type="number" min="1" max="1000" [(ngModel)]="settings.public_data.cod_max_interactive_results" (ngModelChange)="markChanged()" /><span class="settings-help">Maximum structures returned by interactive COD searches.</span></label>
                                <label class="settings-field" for="setting-health-ttl"><span class="settings-label">Provider health cache (seconds)</span><input id="setting-health-ttl" type="number" min="30" max="86400" [(ngModel)]="settings.public_data.health_cache_ttl_seconds" (ngModelChange)="markChanged()" /><span class="settings-help">How long a provider health result remains fresh.</span></label>
                                <label class="settings-field" for="setting-response-limit"><span class="settings-label">Maximum response size (bytes)</span><input id="setting-response-limit" type="number" min="1024" max="268435456" step="1024" [(ngModel)]="settings.public_data.sync_response_size_limit_bytes" (ngModelChange)="markChanged()" /><span class="settings-help">Per-response memory guard, from 1 KiB to 256 MiB.</span></label>
                            </div></div>
                            <div class="settings-subsection"><h4>Sync jobs</h4><div class="settings-grid">
                                <label class="settings-field" for="setting-sync-batch"><span class="settings-label">Records per batch</span><input id="setting-sync-batch" type="number" min="1" max="500" [(ngModel)]="settings.public_data.sync_batch_size" (ngModelChange)="markChanged()" /><span class="settings-help">NIST records processed before progress is checkpointed.</span></label>
                                <label class="settings-field" for="setting-sync-limit"><span class="settings-label">Record limit per sync job</span><input id="setting-sync-limit" type="number" min="1" max="10000" [(ngModel)]="settings.public_data.sync_max_records_per_run" (ngModelChange)="markChanged()" /><span class="settings-help">Maximum records in one NIST, PubChem or COD acquisition.</span></label>
                                <label class="settings-field" for="setting-sync-workers"><span class="settings-label">Concurrent sync jobs</span><input id="setting-sync-workers" type="number" min="1" max="4" [(ngModel)]="settings.public_data.sync_max_concurrent_jobs" (ngModelChange)="markChanged()" /><span class="settings-help">Worker count adjusts while ADSMOD is running.</span></label>
                                <label class="settings-field settings-toggle-field"><span><span class="settings-label">Resume interrupted syncs on startup</span><span class="settings-help">Requeue recoverable public-data jobs after a restart.</span></span><input type="checkbox" [(ngModel)]="settings.public_data.sync_resume_on_startup" (ngModelChange)="markChanged()" /></label>
                            </div></div>
                            <div class="settings-subsection"><h4>Scheduled NIST refresh</h4><div class="settings-grid">
                                <label class="settings-field settings-toggle-field"><span><span class="settings-label">Enable automatic refresh</span><span class="settings-help">Schedule a bounded NIST refresh using the selected collections.</span></span><input type="checkbox" [(ngModel)]="settings.public_data.auto_sync_enabled" (ngModelChange)="markChanged()" /></label>
                                <fieldset class="settings-field settings-field-wide"><legend class="settings-label">NIST collections</legend><span class="settings-help">Select at least one collection when automatic refresh is enabled.</span><div class="settings-choice-row">@for (category of syncCategories; track category) { <label class="settings-choice"><input type="checkbox" [checked]="settings.public_data.auto_sync_categories.includes(category)" (change)="toggleCategory(settings, category, $event)" /><span>{{ categoryLabels[category] }}</span></label> }</div></fieldset>
                                <label class="settings-field" for="setting-sync-interval"><span class="settings-label">Refresh interval (hours)</span><input id="setting-sync-interval" type="number" min="1" max="720" [(ngModel)]="settings.public_data.auto_sync_interval_hours" (ngModelChange)="markChanged()" /><span class="settings-help">Time between scheduled acquisitions.</span></label>
                                <label class="settings-field" for="setting-sync-fraction"><span class="settings-label">Records per refresh <strong>{{ percent(settings.public_data.auto_sync_fraction) }}</strong></span><input id="setting-sync-fraction" type="range" min="0.01" max="1" step="0.01" [(ngModel)]="settings.public_data.auto_sync_fraction" (ngModelChange)="markChanged()" /><span class="settings-help">Fraction of each remote collection to request.</span></label>
                            </div></div>
                        } @else if (activeTab === 'fitting') {
                            <div class="settings-section-heading"><div><p class="settings-eyebrow">Scientific workflows</p><h3>Fitting &amp; jobs</h3><p>Set defaults used by new fitting runs, result ranking and job status updates.</p></div></div>
                            <div class="settings-grid">
                                <label class="settings-field" for="setting-fit-iterations"><span class="settings-label">Default maximum iterations</span><input id="setting-fit-iterations" type="number" min="10" max="1000000000" [(ngModel)]="settings.fitting.default_max_iterations" (ngModelChange)="markChanged()" /><span class="settings-help">Starting value for the fitting workflow.</span></label>
                                <label class="settings-field" for="setting-fit-ceiling"><span class="settings-label">Maximum iteration ceiling</span><input id="setting-fit-ceiling" type="number" min="10" max="1000000000" [(ngModel)]="settings.fitting.max_iterations_upper_bound" (ngModelChange)="markChanged()" /><span class="settings-help">Upper bound enforced when a fit is submitted.</span></label>
                                <label class="settings-field" for="setting-param-min"><span class="settings-label">Default parameter minimum</span><input id="setting-param-min" type="number" min="0" step="any" [(ngModel)]="settings.fitting.default_parameter_min" (ngModelChange)="markChanged()" /><span class="settings-help">Lower bound suggested for model parameters.</span></label>
                                <label class="settings-field" for="setting-param-initial"><span class="settings-label">Default parameter initial value</span><input id="setting-param-initial" type="number" min="0" step="any" [(ngModel)]="settings.fitting.default_parameter_initial" (ngModelChange)="markChanged()" /><span class="settings-help">Starting estimate used by model parameter controls.</span></label>
                                <label class="settings-field" for="setting-param-max"><span class="settings-label">Default parameter maximum</span><input id="setting-param-max" type="number" min="0" step="any" [(ngModel)]="settings.fitting.default_parameter_max" (ngModelChange)="markChanged()" /><span class="settings-help">Upper bound suggested for model parameters.</span></label>
                                <label class="settings-field" for="setting-best-metric"><span class="settings-label">Best model metric</span><select id="setting-best-metric" class="select-input" [(ngModel)]="settings.fitting.best_model_metric" (ngModelChange)="markChanged()">@for (metric of modelMetrics; track metric.value) { <option [value]="metric.value">{{ metric.label }}</option> }</select><span class="settings-help">Ranks successful fits by the selected metric.</span></label>
                                <label class="settings-field" for="setting-polling"><span class="settings-label">Job polling interval (seconds)</span><input id="setting-polling" type="number" min="0.1" max="60" step="0.1" [(ngModel)]="settings.jobs.polling_interval" (ngModelChange)="markChanged()" /><span class="settings-help">Default client polling cadence for new job requests.</span></label>
                            </div>
                        } @else {
                            <div class="settings-section-heading"><div><p class="settings-eyebrow">Machine learning profile</p><h3>Training</h3><p>Defaults are applied to future training jobs. Active training runs keep their captured configuration.</p></div></div>
                            <div class="settings-grid">
                                <label class="settings-field settings-toggle-field"><span><span class="settings-label">Enable JIT compilation</span><span class="settings-help">Compile the model during training when supported by the selected backend.</span></span><input type="checkbox" [(ngModel)]="settings.training.use_jit" (ngModelChange)="markChanged()" /></label>
                                <label class="settings-field" for="setting-jit-backend"><span class="settings-label">JIT backend</span><select id="setting-jit-backend" class="select-input" [(ngModel)]="settings.training.jit_backend" (ngModelChange)="markChanged()"><option value="inductor">Inductor</option><option value="eager">Eager</option><option value="aot_eager">AOT Eager</option><option value="cudagraphs">CUDA Graphs</option></select><span class="settings-help">Torch compile backend used when JIT is enabled.</span></label>
                                <label class="settings-field settings-toggle-field"><span><span class="settings-label">Mixed precision</span><span class="settings-help">Use mixed precision for future model training when the device supports it.</span></span><input type="checkbox" [(ngModel)]="settings.training.use_mixed_precision" (ngModelChange)="markChanged()" /></label>
                                <label class="settings-field" for="setting-workers"><span class="settings-label">Data loader workers</span><input id="setting-workers" type="number" min="0" max="64" [(ngModel)]="settings.training.dataloader_workers" (ngModelChange)="updateWorkerCount(settings, $event)" /><span class="settings-help">Use 0 to load data in the training process.</span></label>
                                <label class="settings-field settings-toggle-field"><span><span class="settings-label">Persistent workers</span><span class="settings-help">Keep loader workers alive between epochs. Requires at least one worker.</span></span><input type="checkbox" [disabled]="settings.training.dataloader_workers === 0" [(ngModel)]="settings.training.persistent_workers" (ngModelChange)="markChanged()" /></label>
                            </div>
                        }
                    </div>
                </div>

                <footer class="settings-actions">
                    <div class="settings-actions-note" aria-live="polite">{{ dirty ? 'Unsaved changes' : 'All changes saved' }}</div>
                    <div class="settings-action-buttons">
                        <button class="button quiet" type="button" [disabled]="saving" (click)="cancel()">Cancel</button>
                        <button class="button secondary" type="button" [disabled]="saving" (click)="reset()">Reset to defaults</button>
                        <button class="button primary" type="button" [disabled]="!dirty || saving || validationErrors.length > 0" (click)="save()">{{ saving ? 'Saving…' : 'Save settings' }}</button>
                    </div>
                </footer>
            }
        </section>
    `,
    styleUrl: './settings-page.component.css',
})
export class SettingsPageComponent {
    private readonly workspace = inject(CoreWorkspaceStore);
    private readonly changeDetector = inject(ChangeDetectorRef);
    protected readonly tabs = [
        { id: 'runtime' as const, label: 'Runtime', mark: '◉' },
        { id: 'import' as const, label: 'Data import', mark: '↥' },
        { id: 'public-data' as const, label: 'Public data', mark: '⇩' },
        { id: 'fitting' as const, label: 'Fitting & jobs', mark: '⌁' },
        { id: 'training' as const, label: 'Training', mark: '✺' },
    ];
    protected readonly extensions: SettingExtension[] = ['.csv', '.xls', '.xlsx'];
    protected readonly syncCategories: SyncCategory[] = ['experiments', 'guest', 'host'];
    protected readonly categoryLabels: Record<SyncCategory, string> = {
        experiments: 'Experiments',
        guest: 'Guest species',
        host: 'Host materials',
    };
    protected readonly modelMetrics: Array<{ value: SettingsModelMetric; label: string }> = [
        { value: 'aicc', label: 'AICc · lower is better' },
        { value: 'aic', label: 'AIC · lower is better' },
        { value: 'bic', label: 'BIC · lower is better' },
        { value: 'sse', label: 'Sum of squared errors · lower is better' },
        { value: 'rmse', label: 'Root mean squared error · lower is better' },
        { value: 'mae', label: 'Mean absolute error · lower is better' },
        { value: 'chi_square', label: 'Chi-square · lower is better' },
        { value: 'r_squared', label: 'R² · higher is better' },
        { value: 'adjusted_r_squared', label: 'Adjusted R² · higher is better' },
    ];

    protected draft: (UserSettings & { pending_restart_fields: string[] }) | null = null;
    protected saved: UserSettings | null = null;
    protected activeTab: SettingsTab = 'runtime';
    protected loading = true;
    protected saving = false;
    protected loadError = '';
    protected errorMessage = '';
    protected statusMessage = '';
    protected dialogMode: 'reset' | 'leave' | null = null;
    private dialogResolver: ((confirmed: boolean) => void) | null = null;
    private dialogOpener: HTMLElement | null = null;

    protected get dirty(): boolean {
        return this.draft !== null && this.saved !== null && JSON.stringify(this.settingsOnly(this.draft)) !== JSON.stringify(this.saved);
    }

    protected get validationErrors(): string[] {
        const value = this.draft;
        if (!value) return [];
        const errors: string[] = [];
        const numberRules: Array<{
            label: string;
            value: number;
            minimum: number;
            maximum?: number;
            integer?: boolean;
            exclusiveMinimum?: boolean;
        }> = [
            { label: 'Backend port', value: value.runtime.backend_port, minimum: 1024, maximum: 65535, integer: true },
            { label: 'Frontend port', value: value.runtime.frontend_port, minimum: 1024, maximum: 65535, integer: true },
            { label: 'Detection confidence cutoff', value: value.datasets.column_detection_cutoff, minimum: 0, maximum: 1 },
            { label: 'Preview row limit', value: value.fitting.preview_row_limit, minimum: 1, maximum: 500, integer: true },
            { label: 'NIST parallel requests', value: value.nist.parallel_tasks, minimum: 1, maximum: 100, integer: true },
            { label: 'Request timeout', value: value.public_data.request_timeout_seconds, minimum: 0, maximum: 120, exclusiveMinimum: true },
            { label: 'Retry attempts', value: value.public_data.retry_attempts, minimum: 1, maximum: 6, integer: true },
            { label: 'Maximum retry backoff', value: value.public_data.retry_backoff_max_seconds, minimum: 0.1, maximum: 120 },
            { label: 'PubChem parallel requests', value: value.public_data.pubchem_parallel_requests, minimum: 1, maximum: 3, integer: true },
            { label: 'COD interactive result limit', value: value.public_data.cod_max_interactive_results, minimum: 1, maximum: 1000, integer: true },
            { label: 'Provider health cache', value: value.public_data.health_cache_ttl_seconds, minimum: 30, maximum: 86400, integer: true },
            { label: 'Response size limit', value: value.public_data.sync_response_size_limit_bytes, minimum: 1024, maximum: 268435456, integer: true },
            { label: 'Sync batch size', value: value.public_data.sync_batch_size, minimum: 1, maximum: 500, integer: true },
            { label: 'Sync record limit', value: value.public_data.sync_max_records_per_run, minimum: 1, maximum: 10000, integer: true },
            { label: 'Concurrent sync jobs', value: value.public_data.sync_max_concurrent_jobs, minimum: 1, maximum: 4, integer: true },
            { label: 'Automatic sync interval', value: value.public_data.auto_sync_interval_hours, minimum: 1, maximum: 720, integer: true },
            { label: 'Automatic sync fraction', value: value.public_data.auto_sync_fraction, minimum: 0, maximum: 1, exclusiveMinimum: true },
            { label: 'Default maximum iterations', value: value.fitting.default_max_iterations, minimum: 10, maximum: 1000000000, integer: true },
            { label: 'Maximum iteration ceiling', value: value.fitting.max_iterations_upper_bound, minimum: 10, maximum: 1000000000, integer: true },
            { label: 'Default parameter minimum', value: value.fitting.default_parameter_min, minimum: 0 },
            { label: 'Default parameter initial value', value: value.fitting.default_parameter_initial, minimum: 0 },
            { label: 'Default parameter maximum', value: value.fitting.default_parameter_max, minimum: 0 },
            { label: 'Job polling interval', value: value.jobs.polling_interval, minimum: 0.1, maximum: 60 },
            { label: 'Data loader workers', value: value.training.dataloader_workers, minimum: 0, maximum: 64, integer: true },
        ];
        for (const rule of numberRules) {
            if (!Number.isFinite(rule.value)
                || (rule.exclusiveMinimum ? rule.value <= rule.minimum : rule.value < rule.minimum)
                || (rule.maximum !== undefined && rule.value > rule.maximum)
                || (rule.integer && !Number.isInteger(rule.value))) {
                errors.push(`${rule.label} must be ${rule.integer ? 'a whole number' : 'a number'} ${rule.exclusiveMinimum ? 'greater than' : 'at least'} ${rule.minimum}${rule.maximum !== undefined ? ` and at most ${rule.maximum}` : ''}.`);
            }
        }
        if (value.runtime.backend_port === value.runtime.frontend_port) errors.push('Backend and frontend ports must be different.');
        if (value.datasets.allowed_extensions.length === 0) errors.push('Select at least one allowed file extension.');
        if (value.public_data.auto_sync_enabled && value.public_data.auto_sync_categories.length === 0) errors.push('Select at least one NIST collection before enabling automatic refresh.');
        if (value.fitting.max_iterations_upper_bound < value.fitting.default_max_iterations) errors.push('The maximum iteration ceiling must be at least the default maximum iterations.');
        if (value.fitting.default_parameter_min >= value.fitting.default_parameter_max) errors.push('The parameter minimum must be less than the maximum.');
        if (value.fitting.default_parameter_initial < value.fitting.default_parameter_min || value.fitting.default_parameter_initial > value.fitting.default_parameter_max) errors.push('The parameter initial value must be between the minimum and maximum.');
        if (value.training.persistent_workers && value.training.dataloader_workers === 0) errors.push('Persistent workers require at least one data loader worker.');
        return errors;
    }

    constructor() {
        void this.load();
    }

    protected async load(): Promise<void> {
        this.loading = true;
        this.loadError = '';
        this.errorMessage = '';
        try {
            this.applyResponse(await fetchSettings());
        } catch (error) {
            this.loadError = error instanceof Error ? error.message : 'Settings could not be loaded.';
        } finally {
            this.loading = false;
            this.changeDetector.detectChanges();
        }
    }

    protected async save(): Promise<void> {
        if (!this.draft || !this.dirty || this.validationErrors.length) return;
        this.saving = true;
        this.errorMessage = '';
        this.statusMessage = '';
        try {
            this.applyResponse(await saveSettings(this.settingsOnly(this.draft)));
            await this.workspace.loadCatalog();
            this.statusMessage = this.draft?.pending_restart_fields.length
                ? 'Settings saved. Restart ADSMOD to apply the pending startup settings.'
                : 'Settings saved and applied to future requests and jobs.';
        } catch (error) {
            this.errorMessage = error instanceof Error ? error.message : 'Settings could not be saved.';
        } finally {
            this.saving = false;
            this.changeDetector.detectChanges();
        }
    }

    protected cancel(): void {
        if (!this.saved || !this.draft) return;
        this.draft = { ...structuredClone(this.saved), pending_restart_fields: [...(this.draft.pending_restart_fields ?? [])] };
        this.errorMessage = '';
        this.statusMessage = 'Unsaved changes discarded.';
    }

    protected async reset(): Promise<void> {
        if (!await this.confirmAction('reset')) return;
        this.saving = true;
        this.errorMessage = '';
        this.statusMessage = '';
        try {
            this.applyResponse(await resetSettings());
            await this.workspace.loadCatalog();
            this.statusMessage = 'All application settings were reset to defaults.';
        } catch (error) {
            this.errorMessage = error instanceof Error ? error.message : 'Settings could not be reset.';
        } finally {
            this.saving = false;
            this.changeDetector.detectChanges();
        }
    }

    protected markChanged(): void {
        this.errorMessage = '';
        this.statusMessage = '';
    }

    protected toggleExtension(settings: UserSettings, extension: SettingExtension, event: Event): void {
        const checked = (event.target as HTMLInputElement).checked;
        const selected = new Set(settings.datasets.allowed_extensions);
        if (checked) selected.add(extension);
        else selected.delete(extension);
        settings.datasets.allowed_extensions = this.extensions.filter((item) => selected.has(item));
        this.markChanged();
    }

    protected toggleCategory(settings: UserSettings, category: SyncCategory, event: Event): void {
        const checked = (event.target as HTMLInputElement).checked;
        const selected = new Set(settings.public_data.auto_sync_categories);
        if (checked) selected.add(category);
        else selected.delete(category);
        settings.public_data.auto_sync_categories = this.syncCategories.filter((item) => selected.has(item));
        this.markChanged();
    }

    protected updateWorkerCount(settings: UserSettings, value: number): void {
        if (Number(value) === 0) settings.training.persistent_workers = false;
        this.markChanged();
    }

    protected handleTabKeydown(event: KeyboardEvent): void {
        const current = TAB_ORDER.indexOf(this.activeTab);
        let next = current;
        if (event.key === 'ArrowRight') next = (current + 1) % TAB_ORDER.length;
        else if (event.key === 'ArrowLeft') next = (current - 1 + TAB_ORDER.length) % TAB_ORDER.length;
        else if (event.key === 'Home') next = 0;
        else if (event.key === 'End') next = TAB_ORDER.length - 1;
        else return;
        event.preventDefault();
        this.activeTab = TAB_ORDER[next];
        queueMicrotask(() => document.getElementById(`settings-tab-${this.activeTab}`)?.focus());
    }

    protected handleDialogKeydown(event: KeyboardEvent): void {
        if (event.key !== 'Tab') return;
        const dialog = event.currentTarget as HTMLElement;
        const focusable = Array.from(dialog.querySelectorAll<HTMLElement>('button:not(:disabled), [href], input:not(:disabled), select:not(:disabled), textarea:not(:disabled), [tabindex]:not([tabindex="-1"])'));
        if (!focusable.length) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (event.shiftKey && document.activeElement === first) {
            event.preventDefault();
            last.focus();
        } else if (!event.shiftKey && document.activeElement === last) {
            event.preventDefault();
            first.focus();
        }
    }

    protected dismissDialog(): void {
        this.finishDialog(false);
    }

    protected confirmDialog(): void {
        this.finishDialog(true);
    }

    protected percent(value: number): string {
        return `${Math.round(Number(value) * 100)}%`;
    }

    protected restartDescription(fields: string[]): string {
        if (fields.includes('public_data.sync_resume_on_startup')) {
            return 'Runtime endpoint changes and recovery behavior take effect after ADSMOD restarts.';
        }
        return 'Runtime host or port changes take effect after ADSMOD restarts.';
    }

    canLeave(): boolean | Promise<boolean> {
        return !this.dirty || this.confirmAction('leave');
    }

    @HostListener('document:keydown', ['$event'])
    protected dismissDialogOnEscape(event: Event): void {
        if ((event as KeyboardEvent).key !== 'Escape') return;
        if (!this.dialogMode) return;
        event.preventDefault();
        this.dismissDialog();
    }

    @HostListener('window:beforeunload', ['$event'])
    protected warnBeforeUnload(event: BeforeUnloadEvent): void {
        if (!this.dirty) return;
        event.preventDefault();
        event.returnValue = '';
    }

    private applyResponse(response: SettingsResponse): void {
        this.saved = structuredClone(response.current);
        this.draft = {
            ...structuredClone(response.current),
            pending_restart_fields: [...response.pending_restart_fields],
        };
    }

    private settingsOnly(value: UserSettings & { pending_restart_fields: string[] }): UserSettings {
        const { pending_restart_fields: _pending, ...settings } = value;
        return settings;
    }

    private confirmAction(mode: 'reset' | 'leave'): Promise<boolean> {
        if (this.dialogResolver) return Promise.resolve(false);
        this.dialogOpener = document.activeElement instanceof HTMLElement ? document.activeElement : null;
        this.dialogMode = mode;
        this.changeDetector.detectChanges();
        queueMicrotask(() => document.getElementById('settings-confirm-primary')?.focus());
        return new Promise((resolve) => {
            this.dialogResolver = resolve;
        });
    }

    private finishDialog(confirmed: boolean): void {
        const mode = this.dialogMode;
        const resolve = this.dialogResolver;
        if (!mode || !resolve) return;
        this.dialogMode = null;
        this.dialogResolver = null;
        this.changeDetector.detectChanges();
        resolve(confirmed);
        if (mode === 'reset' || !confirmed) {
            queueMicrotask(() => this.dialogOpener?.focus());
        }
        this.dialogOpener = null;
    }
}
