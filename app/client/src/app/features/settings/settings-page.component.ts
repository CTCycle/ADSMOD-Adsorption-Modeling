import { CommonModule } from '@angular/common';
import { ChangeDetectorRef, Component, DestroyRef, HostListener, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CoreWorkspaceStore } from '../../core/state/core-workspace.store';
import type { DatabaseSettings, DatabaseSettingsRequest, SettingsModelMetric, SettingsResponse, UserSettings } from '../../models/settings.model';
import { fetchDatabaseSettings, fetchSettings, saveSettings, switchDatabase, testDatabaseConnection } from '../../services/settings.service';

type SettingsTab = 'import' | 'public-data' | 'fitting' | 'training' | 'database';
type SettingExtension = '.csv' | '.xls' | '.xlsx';
type SyncCategory = 'experiments' | 'guest' | 'host';
type DatabaseForm = DatabaseSettingsRequest & { password_configured: boolean };

const TAB_ORDER: readonly SettingsTab[] = ['import', 'public-data', 'fitting', 'training', 'database'];

@Component({
    selector: 'adsmod-settings-page',
    standalone: true,
    imports: [CommonModule, FormsModule],
    template: `
        <section class="settings-page" aria-label="Application settings">
            @if (activeTab === 'database') {
                <div class="settings-workspace">
                    <div class="settings-tabs" role="tablist" aria-label="Settings categories" (keydown)="handleTabKeydown($event)">
                        @for (tab of tabs; track tab.id) {
                            <button class="settings-tab" type="button" role="tab" [id]="'settings-tab-' + tab.id" [attr.aria-selected]="activeTab === tab.id" [attr.aria-controls]="'settings-panel'" [attr.tabindex]="activeTab === tab.id ? 0 : -1" (click)="selectTab(tab.id)">
                                <span class="settings-tab-mark" aria-hidden="true">{{ tab.mark }}</span><span>{{ tab.label }}</span>
                            </button>
                        }
                    </div>
                    <div class="settings-panel" role="tabpanel" id="settings-panel" aria-labelledby="settings-tab-database" tabindex="0">
                        <div class="settings-section-heading">
                            <div><h3>Database</h3><p>Choose where ADSMOD stores operational data and application settings.</p></div>
                            @if (databaseSettings) { <span class="restart-tag" [class.connected-tag]="databaseSettings.connected">{{ databaseSettings.connected ? 'Connected' : 'Unavailable' }}</span> }
                        </div>
                        @if (databaseError) { <div class="settings-message error" role="alert">{{ databaseError }} <button class="button secondary" type="button" (click)="loadDatabase()">Retry</button></div> }
                        @if (databaseMessage) { <div class="settings-message" [class.success]="databaseMessageKind === 'success'" [class.error]="databaseMessageKind === 'error'" role="status">{{ databaseMessage }}</div> }
                        @if (databaseLoading && !databaseForm) {
                            <div class="settings-state-card" role="status">Loading database configuration…</div>
                        } @else if (databaseForm; as database) {
                            <div class="settings-callout"><strong>Independent data stores</strong><span>Switching reloads application settings from the selected database. ADSMOD does not copy settings or operational data between databases. Active jobs and public-data sync runs must finish before switching.</span></div>
                            <div class="settings-grid">
                                <label class="settings-field" for="database-backend"><span class="settings-label">Database backend</span><select id="database-backend" class="select-input" [(ngModel)]="database.backend" (ngModelChange)="clearDatabaseMessage()"><option value="sqlite">SQLite</option><option value="postgres">PostgreSQL</option></select><span class="settings-help">SQLite stores data in a local file. PostgreSQL uses an independent server database.</span></label>
                                @if (database.backend === 'sqlite') {
                                    <label class="settings-field settings-field-wide" for="database-sqlite-path"><span class="settings-label">Database file path</span><input id="database-sqlite-path" type="text" autocomplete="off" [(ngModel)]="database.sqlite_path" (ngModelChange)="clearDatabaseMessage()" /><span class="settings-help">Absolute path or path relative to the ADSMOD data directory.</span></label>
                                    <label class="settings-field" for="database-sqlite-timeout"><span class="settings-label">Connection timeout (seconds)</span><input id="database-sqlite-timeout" type="number" min="1" max="300" step="1" [(ngModel)]="database.connect_timeout" (ngModelChange)="clearDatabaseMessage()" /><span class="settings-help">Maximum wait for SQLite to open the database.</span></label>
                                } @else {
                                    <label class="settings-field" for="database-host"><span class="settings-label">Host</span><input id="database-host" type="text" autocomplete="off" [(ngModel)]="database.host" (ngModelChange)="clearDatabaseMessage()" /></label>
                                    <label class="settings-field" for="database-port"><span class="settings-label">Port</span><input id="database-port" type="number" min="1" max="65535" step="1" [(ngModel)]="database.port" (ngModelChange)="clearDatabaseMessage()" /></label>
                                    <label class="settings-field" for="database-name"><span class="settings-label">Database name</span><input id="database-name" type="text" autocomplete="off" [(ngModel)]="database.database_name" (ngModelChange)="clearDatabaseMessage()" /></label>
                                    <label class="settings-field" for="database-username"><span class="settings-label">Username</span><input id="database-username" type="text" autocomplete="username" [(ngModel)]="database.username" (ngModelChange)="clearDatabaseMessage()" /></label>
                                    <label class="settings-field" for="database-password"><span class="settings-label">Password</span><input id="database-password" type="password" autocomplete="new-password" [(ngModel)]="database.password" (ngModelChange)="clearDatabaseMessage()" [placeholder]="database.password_configured ? 'Configured — leave blank to keep current' : 'Enter password'" /><span class="settings-help">Passwords are never displayed. Leave blank to keep the current password.</span></label>
                                    @if (database.password_configured) { <label class="settings-field settings-toggle-field"><span><span class="settings-label">Clear saved password</span><span class="settings-help">Use only when the PostgreSQL account does not require a password.</span></span><input type="checkbox" [(ngModel)]="database.clear_password" (ngModelChange)="clearDatabaseMessage()" /></label> }
                                    <label class="settings-field settings-toggle-field"><span><span class="settings-label">Require TLS</span><span class="settings-help">Require an encrypted PostgreSQL connection.</span></span><input type="checkbox" [(ngModel)]="database.ssl" (ngModelChange)="clearDatabaseMessage()" /></label>
                                    @if (database.ssl) { <label class="settings-field settings-field-wide" for="database-ssl-ca"><span class="settings-label">TLS certificate authority file</span><input id="database-ssl-ca" type="text" autocomplete="off" [(ngModel)]="database.ssl_ca" (ngModelChange)="clearDatabaseMessage()" /><span class="settings-help">Optional path to a custom CA certificate.</span></label> }
                                    <label class="settings-field" for="database-timeout"><span class="settings-label">Connection timeout (seconds)</span><input id="database-timeout" type="number" min="1" max="300" step="1" [(ngModel)]="database.connect_timeout" (ngModelChange)="clearDatabaseMessage()" /></label>
                                }
                            </div>
                            @if (databaseValidationErrors.length) { <div class="settings-message error" role="alert"><strong>Review the database settings:</strong><ul>@for (message of databaseValidationErrors; track message) { <li>{{ message }}</li> }</ul></div> }
                            <div class="database-actions">
                                <span class="settings-actions-note">Connection details are stored separately from application settings.</span>
                                <div class="settings-action-buttons">
                                    <button class="button secondary" type="button" [disabled]="databaseTesting || databaseSwitching || databaseValidationErrors.length > 0" (click)="testDatabase()">{{ databaseTesting ? 'Testing…' : 'Test connection' }}</button>
                                    <button class="button primary" type="button" [disabled]="databaseTesting || databaseSwitching || databaseValidationErrors.length > 0" (click)="activateDatabase()">{{ databaseSwitching ? 'Switching…' : 'Switch database' }}</button>
                                </div>
                            </div>
                        }
                    </div>
                </div>
            } @else if (loading) {
                <div class="settings-state-card" role="status">Loading saved settings…</div>
                <button class="button secondary" type="button" (click)="selectTab('database')">Database settings</button>
            } @else if (loadError) {
                <div class="settings-message error" role="alert">
                    <span>{{ loadError }}</span>
                    <button class="button secondary" type="button" (click)="load()">Retry</button>
                    <button class="button secondary" type="button" (click)="selectTab('database')">Database settings</button>
                </div>
            } @else if (draft; as settings) {
                @if (settings.pending_restart_fields.length) {
                    <div class="settings-message restart" role="status">
                        <span><strong>Restart required.</strong> {{ restartDescription(settings.pending_restart_fields) }}</span>
                    </div>
                }
                @if (validationErrors.length) {
                    <div class="settings-message error" role="alert">
                        <strong>Correct these settings to save automatically:</strong>
                        <ul>@for (message of validationErrors; track message) { <li>{{ message }}</li> }</ul>
                    </div>
                }
                @if (errorMessage) {
                    <div class="settings-message error" role="alert">
                        <span>{{ errorMessage }}</span>
                        @if (dirty) { <button class="button secondary" type="button" (click)="retryAutosave()">Retry</button> }
                    </div>
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
                                (click)="selectTab(tab.id)"
                            >
                                <span class="settings-tab-mark" aria-hidden="true">{{ tab.mark }}</span>
                                <span>{{ tab.label }}</span>
                            </button>
                        }
                    </div>

                    <div class="settings-panel" role="tabpanel" id="settings-panel" [attr.aria-labelledby]="'settings-tab-' + activeTab" tabindex="0">
                        @if (activeTab === 'import') {
                            <div class="settings-section-heading"><div><h3>Data import</h3><p>Choose accepted spreadsheet formats and how confidently ADSMOD suggests column roles.</p></div></div>
                            <div class="settings-grid">
                                <fieldset class="settings-field settings-field-wide"><legend class="settings-label">Allowed file extensions</legend><span class="settings-help">Only formats supported by the import parser are available.</span><div class="settings-choice-row">@for (extension of extensions; track extension) { <label class="settings-choice"><input type="checkbox" [checked]="settings.datasets.allowed_extensions.includes(extension)" (change)="toggleExtension(settings, extension, $event)" /><span>{{ extension }}</span></label> }</div></fieldset>
                                <label class="settings-field" for="setting-detection-cutoff"><span class="settings-label">Detection confidence cutoff <strong>{{ percent(settings.datasets.column_detection_cutoff) }}</strong></span><input id="setting-detection-cutoff" type="range" min="0" max="1" step="0.01" [(ngModel)]="settings.datasets.column_detection_cutoff" (ngModelChange)="markChanged()" /><span class="settings-help">Column suggestions below this score are marked for manual mapping.</span></label>
                                <label class="settings-field" for="setting-preview-rows"><span class="settings-label">Preview row limit</span><input id="setting-preview-rows" type="number" min="1" max="500" step="1" [(ngModel)]="settings.fitting.preview_row_limit" (ngModelChange)="markChanged()" /><span class="settings-help">Maximum source rows shown in the import preview.</span></label>
                            </div>
                        } @else if (activeTab === 'public-data') {
                            <div class="settings-section-heading"><div><h3>Public data</h3><p>Set request pacing, safe response limits and scheduled NIST refresh behavior.</p></div></div>
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
                            <div class="settings-section-heading"><div><h3>Fitting &amp; jobs</h3><p>Set defaults used by new fitting runs, result ranking and job status updates.</p></div></div>
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
                            <div class="settings-section-heading"><div><h3>Training</h3><p>Defaults are applied to future training jobs. Active training runs keep their captured configuration.</p></div></div>
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
                    <div class="settings-actions-note" role="status" aria-live="polite">
                        {{ saving ? 'Saving changes…' : validationErrors.length ? 'Correct the highlighted values to save.' : dirty ? 'Changes save automatically.' : statusMessage || 'All changes saved.' }}
                    </div>
                    <div class="settings-action-buttons">
                        <button class="button secondary" type="button" (click)="reset()">Reset to defaults</button>
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
        { id: 'import' as const, label: 'Data import', mark: '↥' },
        { id: 'public-data' as const, label: 'Public data', mark: '⇩' },
        { id: 'fitting' as const, label: 'Fitting & jobs', mark: '⌁' },
        { id: 'training' as const, label: 'Training', mark: '✺' },
        { id: 'database' as const, label: 'Database', mark: '▤' },
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
    protected defaults: UserSettings | null = null;
    protected activeTab: SettingsTab = 'import';
    protected loading = true;
    protected saving = false;
    protected loadError = '';
    protected errorMessage = '';
    protected statusMessage = '';
    protected databaseSettings: DatabaseSettings | null = null;
    protected databaseForm: DatabaseForm | null = null;
    protected databaseLoading = true;
    protected databaseTesting = false;
    protected databaseSwitching = false;
    protected databaseError = '';
    protected databaseMessage = '';
    protected databaseMessageKind: 'success' | 'error' = 'success';
    private autosaveTimer: ReturnType<typeof setTimeout> | null = null;
    private flushPromise: Promise<boolean> | null = null;
    private readonly destroyRef = inject(DestroyRef);
    private static readonly AUTOSAVE_DEBOUNCE_MS = 400;

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
        if (value.datasets.allowed_extensions.length === 0) errors.push('Select at least one allowed file extension.');
        if (value.public_data.auto_sync_enabled && value.public_data.auto_sync_categories.length === 0) errors.push('Select at least one NIST collection before enabling automatic refresh.');
        if (value.fitting.max_iterations_upper_bound < value.fitting.default_max_iterations) errors.push('The maximum iteration ceiling must be at least the default maximum iterations.');
        if (value.fitting.default_parameter_min >= value.fitting.default_parameter_max) errors.push('The parameter minimum must be less than the maximum.');
        if (value.fitting.default_parameter_initial < value.fitting.default_parameter_min || value.fitting.default_parameter_initial > value.fitting.default_parameter_max) errors.push('The parameter initial value must be between the minimum and maximum.');
        if (value.training.persistent_workers && value.training.dataloader_workers === 0) errors.push('Persistent workers require at least one data loader worker.');
        return errors;
    }

    protected get databaseValidationErrors(): string[] {
        const form = this.databaseForm;
        if (!form) return [];
        const errors: string[] = [];
        if (form.backend === 'sqlite' && (!form.sqlite_path.trim() || form.sqlite_path.trim() === ':memory:')) errors.push('Enter a persistent SQLite file path. In-memory databases are reserved for tests.');
        if (!Number.isInteger(form.connect_timeout) || form.connect_timeout < 1 || form.connect_timeout > 300) errors.push('Connection timeout must be a whole number from 1 to 300 seconds.');
        if (form.backend === 'postgres') {
            if (!form.host.trim()) errors.push('Enter a PostgreSQL host.');
            if (!form.database_name.trim()) errors.push('Enter a PostgreSQL database name.');
            if (!form.username.trim()) errors.push('Enter a PostgreSQL username.');
            if (!Number.isInteger(form.port) || form.port < 1 || form.port > 65535) errors.push('PostgreSQL port must be a whole number from 1 to 65535.');
        }
        return errors;
    }

    constructor() {
        this.destroyRef.onDestroy(() => this.clearAutosaveTimer());
        void this.load();
        void this.loadDatabase();
    }

    protected selectTab(tab: SettingsTab): void {
        this.activeTab = tab;
        if (tab === 'database' && !this.databaseForm && !this.databaseLoading) void this.loadDatabase();
    }

    protected async loadDatabase(): Promise<void> {
        this.databaseLoading = true;
        this.databaseError = '';
        try {
            const current = await fetchDatabaseSettings();
            this.databaseSettings = current;
            this.databaseForm = {
                backend: current.backend,
                sqlite_path: current.sqlite_path ?? 'data/database.db',
                engine: 'postgresql+psycopg',
                host: current.host ?? '',
                port: current.port ?? 5432,
                database_name: current.database_name ?? '',
                username: current.username ?? '',
                password: '',
                clear_password: false,
                ssl: current.ssl,
                ssl_ca: current.ssl_ca ?? '',
                connect_timeout: current.connect_timeout,
                password_configured: current.password_configured,
            };
        } catch (error) {
            this.databaseError = error instanceof Error ? error.message : 'Database configuration could not be loaded.';
        } finally {
            this.databaseLoading = false;
            this.changeDetector.detectChanges();
        }
    }

    protected clearDatabaseMessage(): void {
        this.databaseMessage = '';
    }

    protected async testDatabase(): Promise<void> {
        if (!this.databaseForm || this.databaseValidationErrors.length) return;
        this.databaseTesting = true;
        this.databaseMessage = '';
        try {
            const result = await testDatabaseConnection(this.databaseRequest(this.databaseForm));
            this.databaseMessage = result.message;
            this.databaseMessageKind = 'success';
        } catch (error) {
            this.databaseMessage = error instanceof Error ? error.message : 'Connection test failed.';
            this.databaseMessageKind = 'error';
        } finally {
            this.databaseTesting = false;
            this.changeDetector.detectChanges();
        }
    }

    protected async activateDatabase(): Promise<void> {
        if (!this.databaseForm || this.databaseValidationErrors.length) return;
        this.databaseSwitching = true;
        this.databaseMessage = '';
        try {
            if (!await this.flushPendingChanges()) {
                this.databaseMessage = this.validationErrors.length
                    ? 'Correct application settings before switching databases.'
                    : 'Application settings could not be saved. Retry before switching databases.';
                this.databaseMessageKind = 'error';
                return;
            }
            this.databaseSettings = await switchDatabase(this.databaseRequest(this.databaseForm));
            window.location.reload();
        } catch (error) {
            this.databaseMessage = error instanceof Error ? error.message : 'Database switch failed.';
            this.databaseMessageKind = 'error';
        } finally {
            this.databaseSwitching = false;
            this.changeDetector.detectChanges();
        }
    }

    private databaseRequest(form: DatabaseForm): DatabaseSettingsRequest {
        const { password_configured: _passwordConfigured, ...request } = form;
        return request;
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

    protected reset(): void {
        if (!this.defaults || !this.draft) return;
        this.draft = {
            ...structuredClone(this.defaults),
            pending_restart_fields: [...this.draft.pending_restart_fields],
        };
        this.errorMessage = '';
        this.markChanged();
    }

    protected retryAutosave(): void {
        this.errorMessage = '';
        void this.flushPendingChanges();
    }

    protected markChanged(): void {
        this.errorMessage = '';
        this.statusMessage = '';
        this.scheduleAutosave();
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

    protected percent(value: number): string {
        return `${Math.round(Number(value) * 100)}%`;
    }

    protected restartDescription(fields: string[]): string {
        if (fields.includes('public_data.sync_resume_on_startup')) {
            return 'Interrupted public-data sync recovery changes take effect after ADSMOD restarts.';
        }
        return 'Restart ADSMOD to apply the pending startup change.';
    }

    flushPendingChanges(): Promise<boolean> {
        this.clearAutosaveTimer();
        if (!this.draft || !this.saved || !this.dirty) return Promise.resolve(true);
        if (this.validationErrors.length) {
            this.changeDetector.detectChanges();
            return Promise.resolve(false);
        }
        if (this.flushPromise) {
            return this.flushPromise.then((saved) => {
                if (!saved) return false;
                return this.dirty ? this.flushPendingChanges() : true;
            });
        }

        const operation = this.saveLatestChanges();
        this.flushPromise = operation;
        return operation.finally(() => {
            if (this.flushPromise === operation) this.flushPromise = null;
        });
    }

    canLeave(): Promise<boolean> {
        return this.flushPendingChanges();
    }

    @HostListener('window:pagehide')
    protected persistOnPageHide(): void {
        if (!this.draft || !this.dirty || this.validationErrors.length) return;
        this.clearAutosaveTimer();
        void saveSettings(this.settingsOnly(this.draft), { keepalive: true }).catch(() => undefined);
    }

    private applyResponse(response: SettingsResponse): void {
        this.saved = structuredClone(response.current);
        this.defaults = structuredClone(response.defaults);
        this.draft = {
            ...structuredClone(response.current),
            pending_restart_fields: [...response.pending_restart_fields],
        };
    }

    private scheduleAutosave(): void {
        this.clearAutosaveTimer();
        if (!this.draft || !this.saved || this.validationErrors.length) return;
        this.autosaveTimer = setTimeout(() => {
            this.autosaveTimer = null;
            void this.flushPendingChanges();
        }, SettingsPageComponent.AUTOSAVE_DEBOUNCE_MS);
    }

    private clearAutosaveTimer(): void {
        if (this.autosaveTimer === null) return;
        clearTimeout(this.autosaveTimer);
        this.autosaveTimer = null;
    }

    private async saveLatestChanges(): Promise<boolean> {
        while (this.dirty) {
            if (!this.draft || this.validationErrors.length) {
                this.changeDetector.detectChanges();
                return false;
            }
            const snapshot = structuredClone(this.settingsOnly(this.draft));
            const refreshCatalog = JSON.stringify(this.saved?.datasets.allowed_extensions)
                !== JSON.stringify(snapshot.datasets.allowed_extensions);
            this.saving = true;
            this.errorMessage = '';
            this.statusMessage = '';
            try {
                const response = await saveSettings(snapshot);
                const draftMatchesSnapshot = this.draft !== null
                    && JSON.stringify(this.settingsOnly(this.draft))
                    === JSON.stringify(snapshot);
                this.saved = structuredClone(response.current);
                if (draftMatchesSnapshot) {
                    this.draft = {
                        ...structuredClone(response.current),
                        pending_restart_fields: [...response.pending_restart_fields],
                    };
                    this.statusMessage = response.pending_restart_fields.length
                        ? 'Saved. Restart ADSMOD to apply the pending startup change.'
                        : 'Saved and applied.';
                    if (refreshCatalog) {
                        try {
                            await this.workspace.loadCatalog();
                        } catch {
                            this.errorMessage = 'Settings were saved, but the workspace catalog could not refresh. Reload ADSMOD to update available file formats.';
                        }
                    }
                } else if (this.draft) {
                    this.draft.pending_restart_fields = [...response.pending_restart_fields];
                }
            } catch (error) {
                if (this.draft
                    && JSON.stringify(this.settingsOnly(this.draft)) !== JSON.stringify(snapshot)) {
                    continue;
                }
                this.errorMessage = error instanceof Error
                    ? 'Changes could not be saved: ' + error.message
                    : 'Changes could not be saved. Check the connection and retry.';
                this.statusMessage = '';
                return false;
            } finally {
                this.saving = false;
                this.changeDetector.detectChanges();
            }
        }
        return true;
    }

    private settingsOnly(value: UserSettings & { pending_restart_fields: string[] }): UserSettings {
        const { pending_restart_fields: _pending, ...settings } = value;
        return settings;
    }

}
