import { Component, computed, input, output, signal } from '@angular/core';
import { FormControl, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import type { TorchCompileBackend, TrainingConfig } from '../../../models/training.model';
import { CheckboxComponent } from '../../../shared/components/checkbox/checkbox.component';
import { NumberInputComponent } from '../../../shared/components/number-input/number-input.component';
import { SwitchComponent } from '../../../shared/components/switch/switch.component';
import { WizardNavigationFooterComponent } from './wizard-navigation-footer.component';
import { WizardProgressIndicatorComponent } from './wizard-progress-indicator.component';

const TORCH_COMPILE_BACKENDS: readonly TorchCompileBackend[] = ['inductor', 'cudagraphs', 'aot_eager', 'eager'];
const GPU_DEVICE_OPTIONS = Array.from({ length: 16 }, (_, index) => index);
const LAST_PAGE_INDEX = 4;

@Component({
    selector: 'adsmod-new-training-wizard',
    standalone: true,
    imports: [
        ReactiveFormsModule,
        CheckboxComponent,
        NumberInputComponent,
        SwitchComponent,
        WizardNavigationFooterComponent,
        WizardProgressIndicatorComponent,
    ],
    template: `
        <div class="modal-backdrop" role="dialog" aria-modal="true" aria-labelledby="new-training-wizard-title">
            <div class="wizard-modal">
                <div class="wizard-header">
                    <h4 id="new-training-wizard-title">New Training Wizard</h4>
                    <p>Configure training settings for dataset: <strong>{{ selectedDatasetLabel() }}</strong></p>
                    <adsmod-wizard-progress-indicator [currentPage]="currentPage()" [totalPages]="LAST_PAGE_INDEX + 1" />
                </div>

                <div class="wizard-body">
                    @if (currentPage() === 0) {
                        <div class="wizard-page">
                            <div class="wizard-card">
                                <div class="wizard-card-header"><span class="wizard-card-icon">📊</span><span>Dataset Configuration</span></div>
                                <p class="wizard-card-description">Control how the training dataset is shuffled during training.</p>
                                <div class="wizard-card-body">
                                    <div class="wizard-toggle-row wizard-toggle-row-aligned">
                                        <div class="wizard-toggle-control">
                                            <label>Shuffle Buffered</label>
                                            <div class="wizard-toggle-switch">
                                                <adsmod-switch [checked]="shuffleDataset()" (checkedChange)="shuffleDatasetControl.setValue($event)" />
                                            </div>
                                        </div>
                                        @if (shuffleDataset()) {
                                            <div class="wizard-inline-number-field">
                                                <adsmod-number-input label="Max Buffer Size" [value]="maxBufferSize()" [min]="1" [max]="1000000" [step]="1" [precision]="0" (valueChange)="maxBufferSizeControl.setValue($event)" />
                                            </div>
                                        }
                                    </div>
                                </div>
                            </div>
                        </div>
                    }

                    @if (currentPage() === 1) {
                        <div class="wizard-page">
                            <div class="wizard-card">
                                <div class="wizard-card-header"><span class="wizard-card-icon">🧠</span><span>Model Configuration</span></div>
                                <p class="wizard-card-description">Define the architecture and embedding dimensions for the SCADS model family.</p>
                                <div class="wizard-card-body">
                                    <div class="wizard-settings-grid">
                                        <adsmod-number-input label="Encoders" [value]="numEncoders()" [min]="1" [max]="12" [step]="1" [precision]="0" (valueChange)="numEncodersControl.setValue($event)" />
                                        <adsmod-number-input label="Attention Heads" [value]="numAttentionHeads()" [min]="1" [max]="16" [step]="1" [precision]="0" (valueChange)="numAttentionHeadsControl.setValue($event)" />
                                        <adsmod-number-input label="Embedding Dims" [value]="molecularEmbeddingSize()" [min]="64" [max]="1024" [step]="64" [precision]="0" (valueChange)="molecularEmbeddingSizeControl.setValue($event)" />
                                        <adsmod-number-input label="Dropout Rate" [value]="dropoutRate()" [min]="0" [max]="0.5" [step]="0.05" [precision]="2" (valueChange)="dropoutRateControl.setValue($event)" />
                                        <div style="min-width: 160px; grid-column: span 2; width: 100%;">
                                            <label class="field-label">Model Type</label>
                                            <select class="select-input" style="width: 100%;" [formControl]="selectedModelControl">
                                                <option value="SCADS Series">SCADS Series</option>
                                                <option value="SCADS Atomic">SCADS Atomic</option>
                                            </select>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    }

                    @if (currentPage() === 2) {
                        <div class="wizard-page">
                            <div class="wizard-card">
                                <div class="wizard-card-header"><span class="wizard-card-icon">⚙️</span><span>Training Configuration</span></div>
                                <p class="wizard-card-description">Define the training schedule, checkpointing, and learning rate behavior.</p>
                                <div class="wizard-card-body">
                                    <div class="wizard-settings-grid">
                                        <adsmod-number-input label="Epochs" [value]="epochs()" [min]="1" [max]="500" [step]="1" [precision]="0" (valueChange)="epochsControl.setValue($event)" />
                                        <adsmod-number-input label="Batch Size" [value]="batchSize()" [min]="1" [max]="256" [step]="1" [precision]="0" (valueChange)="batchSizeControl.setValue($event)" />
                                        <div class="wizard-toggle-column">
                                            <adsmod-checkbox label="Save Checkpoints" [checked]="saveCheckpoints()" (checkedChange)="saveCheckpointsControl.setValue($event)" />
                                            <adsmod-checkbox label="LR Scheduler" [checked]="useLrScheduler()" (checkedChange)="useLrSchedulerControl.setValue($event)" />
                                        </div>
                                    </div>
                                    @if (useLrScheduler()) {
                                        <div class="wizard-settings-grid wizard-settings-grid-tight">
                                            <adsmod-number-input label="Initial LR" [value]="initialLr()" [min]="0.000001" [max]="0.01" [step]="0.00001" [precision]="6" (valueChange)="initialLrControl.setValue($event)" />
                                            <adsmod-number-input label="Target LR" [value]="targetLr()" [min]="0.0000001" [max]="0.001" [step]="0.000001" [precision]="7" (valueChange)="targetLrControl.setValue($event)" />
                                            <adsmod-number-input label="Constant Steps" [value]="constantSteps()" [min]="0" [max]="50" [step]="1" [precision]="0" (valueChange)="constantStepsControl.setValue($event)" />
                                            <adsmod-number-input label="Decay Steps" [value]="decaySteps()" [min]="1" [max]="100" [step]="1" [precision]="0" (valueChange)="decayStepsControl.setValue($event)" />
                                        </div>
                                    }
                                </div>
                            </div>
                        </div>
                    }

                    @if (currentPage() === 3) {
                        <div class="wizard-page">
                            <div class="wizard-card">
                                <div class="wizard-card-header"><span class="wizard-card-icon">🖥️</span><span>Device Controls</span></div>
                                <p class="wizard-card-description">Configure data loading and runtime acceleration options.</p>
                                <div class="wizard-card-body">
                                    <div class="wizard-device-layout">
                                        <div class="wizard-device-column wizard-device-column-left">
                                            <div class="wizard-device-toggle-compact">
                                                <adsmod-checkbox label="Pin Memory" [checked]="pinMemory()" (checkedChange)="pinMemoryControl.setValue($event)" />
                                            </div>
                                            <div class="wizard-device-toggle-compact wizard-device-toggle-compact-spaced">
                                                <adsmod-checkbox label="Mixed Precision" [checked]="useMixedPrecision()" (checkedChange)="useMixedPrecisionControl.setValue($event)" />
                                            </div>
                                            <div class="wizard-compact-field">
                                                <label class="field-label" for="dataloader-workers">Dataloader Workers</label>
                                                <input id="dataloader-workers" class="wizard-compact-input" type="number" [formControl]="dataloaderWorkersControl" min="0" max="64" step="1" />
                                            </div>
                                            <div class="wizard-compact-field">
                                                <label class="field-label" for="prefetch-factor">Prefetch Factor</label>
                                                <input id="prefetch-factor" class="wizard-compact-input" type="number" [formControl]="prefetchFactorControl" min="1" max="32" step="1" />
                                            </div>
                                        </div>
                                        <div class="wizard-device-column wizard-device-column-right">
                                            <div class="wizard-device-option-card">
                                                <h5 class="wizard-device-option-title">Torch Compile</h5>
                                                <p class="wizard-device-option-description">Enable torch.compile to optimize runtime graph execution.</p>
                                                <div class="wizard-device-option-controls">
                                                    <adsmod-checkbox label="Torch Compile" [checked]="useJit()" (checkedChange)="useJitControl.setValue($event)" />
                                                    <div class="wizard-device-option-dropdown">
                                                        <label class="field-label wizard-inline-label" for="torch-compile-backend">Backend</label>
                                                        <select id="torch-compile-backend" class="select-input wizard-inline-select" [formControl]="jitBackendControl" [disabled]="!useJit()">
                                                            @for (backend of torchCompileBackends; track backend) {
                                                                <option [value]="backend">{{ backend }}</option>
                                                            }
                                                        </select>
                                                    </div>
                                                </div>
                                            </div>

                                            <div class="wizard-device-option-card">
                                                <h5 class="wizard-device-option-title">Enable GPU</h5>
                                                <p class="wizard-device-option-description">Run training on CUDA and choose the target GPU device index.</p>
                                                <div class="wizard-device-option-controls">
                                                    <adsmod-checkbox label="Enable GPU" [checked]="useDeviceGpu()" (checkedChange)="useDeviceGpuControl.setValue($event)" />
                                                    <div class="wizard-device-option-dropdown">
                                                        <label class="field-label wizard-inline-label" for="gpu-device-id">Device</label>
                                                        <select id="gpu-device-id" class="select-input wizard-inline-select" [formControl]="deviceIdControl" [disabled]="!useDeviceGpu()">
                                                            @for (deviceId of gpuDeviceOptions; track deviceId) {
                                                                <option [value]="deviceId">{{ deviceId }}</option>
                                                            }
                                                        </select>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    }

                    @if (currentPage() === 4) {
                        <div class="wizard-page">
                            <div class="wizard-card" style="margin-bottom: 1rem; border: 1px solid var(--primary-200);">
                                <div class="wizard-card-header"><span class="wizard-card-icon">🏷️</span><span>Training Name</span></div>
                                <div class="wizard-card-body">
                                    <div style="padding: 0.5rem 0;">
                                        <label class="field-label" style="margin-bottom: 0.5rem; display: block;">Custom Name (Optional)</label>
                                        <input type="text" [formControl]="customNameControl" placeholder="e.g. Experiment_A" class="number-input-field" style="width: 100%; text-align: left; padding: 0.5rem 0.75rem; font-size: 0.95rem; border-radius: 8px; border: 1px solid var(--slate-300); height: auto;" />
                                    </div>
                                </div>
                            </div>

                            <div class="wizard-summary">
                                <div class="wizard-summary-section">
                                    <h5>Selected Dataset</h5>
                                    <div class="wizard-summary-grid">
                                        <span>Dataset</span><strong>{{ selectedDatasetLabel() }}</strong>
                                    </div>
                                </div>
                                <div class="wizard-summary-section">
                                    <h5>Dataset Configuration</h5>
                                    <div class="wizard-summary-grid">
                                        <span>Shuffle buffered</span><strong>{{ shuffleDataset() ? 'Enabled' : 'Disabled' }}</strong>
                                        @if (shuffleDataset()) {
                                            <span>Max buffer size</span><strong>{{ maxBufferSize() }}</strong>
                                        }
                                    </div>
                                </div>
                                <div class="wizard-summary-section">
                                    <h5>Model Configuration</h5>
                                    <div class="wizard-summary-grid">
                                        <span>Encoders</span><strong>{{ numEncoders() }}</strong>
                                        <span>Attention heads</span><strong>{{ numAttentionHeads() }}</strong>
                                        <span>Embedding dims</span><strong>{{ molecularEmbeddingSize() }}</strong>
                                        <span>Dropout rate</span><strong>{{ dropoutRate() }}</strong>
                                        <span>Model type</span><strong>{{ selectedModel() }}</strong>
                                    </div>
                                </div>
                                <div class="wizard-summary-section">
                                    <h5>Training Configuration</h5>
                                    <div class="wizard-summary-grid">
                                        <span>Epochs</span><strong>{{ epochs() }}</strong>
                                        <span>Batch size</span><strong>{{ batchSize() }}</strong>
                                        <span>Save checkpoints</span><strong>{{ saveCheckpoints() ? 'Enabled' : 'Disabled' }}</strong>
                                        <span>LR scheduler</span><strong>{{ useLrScheduler() ? 'Enabled' : 'Disabled' }}</strong>
                                        @if (useLrScheduler()) {
                                            <span>Initial LR</span><strong>{{ initialLr() }}</strong>
                                            <span>Target LR</span><strong>{{ targetLr() }}</strong>
                                            <span>Constant steps</span><strong>{{ constantSteps() }}</strong>
                                            <span>Decay steps</span><strong>{{ decaySteps() }}</strong>
                                        }
                                    </div>
                                </div>
                                <div class="wizard-summary-section">
                                    <h5>Device Controls</h5>
                                    <div class="wizard-summary-grid">
                                        <span>Dataloader workers</span><strong>{{ dataloaderWorkers() }}</strong>
                                        <span>Prefetch factor</span><strong>{{ prefetchFactor() }}</strong>
                                        <span>Pin memory</span><strong>{{ pinMemory() ? 'Enabled' : 'Disabled' }}</strong>
                                        <span>GPU</span><strong>{{ useDeviceGpu() ? 'Enabled' : 'Disabled' }}</strong>
                                        @if (useDeviceGpu()) {
                                            <span>GPU device ID</span><strong>{{ deviceId() }}</strong>
                                        }
                                        <span>Mixed precision</span><strong>{{ useMixedPrecision() ? 'Enabled' : 'Disabled' }}</strong>
                                        <span>Torch compile</span><strong>{{ useJit() ? 'Enabled' : 'Disabled' }}</strong>
                                        @if (useJit()) {
                                            <span>Compile backend</span><strong>{{ jitBackend() }}</strong>
                                        }
                                    </div>
                                </div>
                            </div>
                        </div>
                    }
                </div>

                <adsmod-wizard-navigation-footer
                    [isLoading]="isLoading()"
                    [isFirstPage]="currentPage() === 0"
                    [isLastPage]="currentPage() === LAST_PAGE_INDEX"
                    confirmIdleLabel="Confirm Training"
                    confirmLoadingLabel="Starting..."
                    (closed)="closed.emit()"
                    (next)="goToNextPage()"
                    (previous)="goToPreviousPage()"
                    (confirmed)="confirm()"
                />
            </div>
        </div>
    `,
})
export class NewTrainingWizardComponent {
    readonly selectedDatasetLabel = input.required<string>();
    readonly initialConfig = input.required<TrainingConfig>();
    readonly isLoading = input(false);
    readonly closed = output<void>();
    readonly confirmed = output<TrainingConfig>();
    protected readonly LAST_PAGE_INDEX = LAST_PAGE_INDEX;
    protected readonly torchCompileBackends = TORCH_COMPILE_BACKENDS;
    protected readonly gpuDeviceOptions = GPU_DEVICE_OPTIONS;
    protected readonly currentPage = signal(0);

    protected readonly shuffleDatasetControl = new FormControl(true, { nonNullable: true });
    protected readonly maxBufferSizeControl = new FormControl(256, { nonNullable: true, validators: [Validators.min(1)] });
    protected readonly selectedModelControl = new FormControl<'SCADS Series' | 'SCADS Atomic'>('SCADS Series', { nonNullable: true });
    protected readonly dropoutRateControl = new FormControl(0.1, { nonNullable: true });
    protected readonly numAttentionHeadsControl = new FormControl(2, { nonNullable: true, validators: [Validators.min(1)] });
    protected readonly numEncodersControl = new FormControl(2, { nonNullable: true, validators: [Validators.min(1)] });
    protected readonly molecularEmbeddingSizeControl = new FormControl(64, { nonNullable: true, validators: [Validators.min(64)] });
    protected readonly epochsControl = new FormControl(2, { nonNullable: true, validators: [Validators.min(1)] });
    protected readonly batchSizeControl = new FormControl(16, { nonNullable: true, validators: [Validators.min(1)] });
    protected readonly saveCheckpointsControl = new FormControl(false, { nonNullable: true });
    protected readonly useLrSchedulerControl = new FormControl(false, { nonNullable: true });
    protected readonly initialLrControl = new FormControl(1e-4, { nonNullable: true });
    protected readonly targetLrControl = new FormControl(1e-5, { nonNullable: true });
    protected readonly constantStepsControl = new FormControl(5, { nonNullable: true });
    protected readonly decayStepsControl = new FormControl(10, { nonNullable: true });
    protected readonly pinMemoryControl = new FormControl(true, { nonNullable: true });
    protected readonly useMixedPrecisionControl = new FormControl(false, { nonNullable: true });
    protected readonly dataloaderWorkersControl = new FormControl(0, { nonNullable: true });
    protected readonly prefetchFactorControl = new FormControl(1, { nonNullable: true, validators: [Validators.min(1)] });
    protected readonly useJitControl = new FormControl(false, { nonNullable: true });
    protected readonly jitBackendControl = new FormControl<TorchCompileBackend>('inductor', { nonNullable: true });
    protected readonly useDeviceGpuControl = new FormControl(true, { nonNullable: true });
    protected readonly deviceIdControl = new FormControl(0, { nonNullable: true });
    protected readonly customNameControl = new FormControl('', { nonNullable: true });
    protected readonly form = new FormGroup({
        shuffle_dataset: this.shuffleDatasetControl,
        max_buffer_size: this.maxBufferSizeControl,
        selected_model: this.selectedModelControl,
        dropout_rate: this.dropoutRateControl,
        num_attention_heads: this.numAttentionHeadsControl,
        num_encoders: this.numEncodersControl,
        molecular_embedding_size: this.molecularEmbeddingSizeControl,
        epochs: this.epochsControl,
        batch_size: this.batchSizeControl,
        save_checkpoints: this.saveCheckpointsControl,
        use_lr_scheduler: this.useLrSchedulerControl,
        initial_lr: this.initialLrControl,
        target_lr: this.targetLrControl,
        constant_steps: this.constantStepsControl,
        decay_steps: this.decayStepsControl,
        pin_memory: this.pinMemoryControl,
        use_mixed_precision: this.useMixedPrecisionControl,
        dataloader_workers: this.dataloaderWorkersControl,
        prefetch_factor: this.prefetchFactorControl,
        use_jit: this.useJitControl,
        jit_backend: this.jitBackendControl,
        use_device_GPU: this.useDeviceGpuControl,
        device_ID: this.deviceIdControl,
        custom_name: this.customNameControl,
    });

    constructor() {
        queueMicrotask(() => this.patchInitialConfig());
    }

    protected readonly shuffleDataset = computed(() => this.shuffleDatasetControl.value);
    protected readonly maxBufferSize = computed(() => this.maxBufferSizeControl.value);
    protected readonly selectedModel = computed(() => this.selectedModelControl.value);
    protected readonly dropoutRate = computed(() => this.dropoutRateControl.value);
    protected readonly numAttentionHeads = computed(() => this.numAttentionHeadsControl.value);
    protected readonly numEncoders = computed(() => this.numEncodersControl.value);
    protected readonly molecularEmbeddingSize = computed(() => this.molecularEmbeddingSizeControl.value);
    protected readonly epochs = computed(() => this.epochsControl.value);
    protected readonly batchSize = computed(() => this.batchSizeControl.value);
    protected readonly saveCheckpoints = computed(() => this.saveCheckpointsControl.value);
    protected readonly useLrScheduler = computed(() => this.useLrSchedulerControl.value);
    protected readonly initialLr = computed(() => this.initialLrControl.value);
    protected readonly targetLr = computed(() => this.targetLrControl.value);
    protected readonly constantSteps = computed(() => this.constantStepsControl.value);
    protected readonly decaySteps = computed(() => this.decayStepsControl.value);
    protected readonly pinMemory = computed(() => this.pinMemoryControl.value);
    protected readonly useMixedPrecision = computed(() => this.useMixedPrecisionControl.value);
    protected readonly dataloaderWorkers = computed(() => this.dataloaderWorkersControl.value);
    protected readonly prefetchFactor = computed(() => this.prefetchFactorControl.value);
    protected readonly useJit = computed(() => this.useJitControl.value);
    protected readonly jitBackend = computed(() => this.jitBackendControl.value);
    protected readonly useDeviceGpu = computed(() => this.useDeviceGpuControl.value);
    protected readonly deviceId = computed(() => this.deviceIdControl.value);

    protected goToNextPage(): void {
        this.currentPage.update((page) => Math.min(LAST_PAGE_INDEX, page + 1));
    }

    protected goToPreviousPage(): void {
        this.currentPage.update((page) => Math.max(0, page - 1));
    }

    protected confirm(): void {
        const config = this.initialConfig();
        this.confirmed.emit({
            ...config,
            ...this.form.getRawValue(),
            dataset_label: this.selectedDatasetLabel(),
        });
    }

    private patchInitialConfig(): void {
        this.form.patchValue(this.initialConfig());
    }
}
