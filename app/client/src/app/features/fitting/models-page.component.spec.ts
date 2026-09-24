import { TestBed } from '@angular/core/testing';
import { provideRouter } from '@angular/router';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { CoreWorkspaceStore } from '../../core/state/core-workspace.store';
import type { FittingResponse } from '../../models/fitting.model';
import { ModelsPageComponent } from './models-page.component';

describe('ModelsPageComponent', () => {
    const fetchMock = vi.fn();

    beforeEach(async () => {
        TestBed.resetTestingModule();
        fetchMock.mockReset();
        fetchMock.mockImplementation(async (input: RequestInfo | URL) => {
            const url = String(input);
            if (url.endsWith('/datasets')) {
                return {
                    ok: true,
                    status: 200,
                    json: async () => ({
                        datasets: [
                            {
                                id: 1,
                                name: 'single-experiment dataset',
                                source: 'uploaded',
                                created_at: '2026-08-13T00:00:00Z',
                                experiment_count: 1,
                                observation_count: 4,
                                tags: [],
                                description: '',
                            },
                        ],
                    }),
                };
            }
            if (url.includes('/datasets/1/experiments')) {
                return {
                    ok: true,
                    status: 200,
                    json: async () => ({
                        experiments: [
                            {
                                id: 42,
                                dataset_id: 1,
                                external_key: 'qa-smiles-298',
                                name: 'qa-smiles-298',
                                adsorbent: 'Activated carbon',
                                adsorbates: ['CO2'],
                                temperature_k: 298.15,
                                pressure_basis: 'absolute',
                                observation_count: 4,
                                fitting_eligible: true,
                                ineligibility_reason: null,
                            },
                        ],
                    }),
                };
            }
            if (url.includes('/fitting/models')) {
                return {
                    ok: true,
                    status: 200,
                    json: async () => ({
                        status: 'success',
                        pressure_unit: 'bar',
                        uptake_unit: 'mmol/g',
                        models: [],
                    }),
                };
            }
            throw new Error(`Unhandled URL ${url}`);
        });
        vi.stubGlobal('fetch', fetchMock);
        await TestBed.configureTestingModule({
            imports: [ModelsPageComponent],
            providers: [provideRouter([])],
        }).compileComponents();
    });

    afterEach(() => {
        vi.unstubAllGlobals();
    });

    it('shows the sole experiment selected after choosing its dataset', async () => {
        const fixture = TestBed.createComponent(ModelsPageComponent);
        fixture.detectChanges();
        await fixture.whenStable();
        fixture.detectChanges();

        const root = fixture.nativeElement as HTMLElement;
        const datasetSelect = root.querySelector<HTMLSelectElement>('#fitting-dataset-control');
        expect(datasetSelect).not.toBeNull();
        datasetSelect!.value = '1';
        datasetSelect!.dispatchEvent(new Event('change'));
        await fixture.whenStable();
        fixture.detectChanges();

        const experimentSelect = root.querySelector<HTMLSelectElement>('#fitting-experiment-control');
        expect(TestBed.inject(CoreWorkspaceStore).selectedExperimentId()).toBe(42);
        expect(experimentSelect?.value).toBe('42');
        expect(experimentSelect?.selectedOptions[0]?.textContent).toContain('qa-smiles-298');
    });

    it('renders returned fitting metrics and identifies the best model', async () => {
        const fixture = TestBed.createComponent(ModelsPageComponent);
        fixture.detectChanges();
        await fixture.whenStable();

        const fittingResult: FittingResponse = {
            status: 'success',
            run_id: 7,
            dataset_id: 1,
            isotherm_id: 42,
            dataset_name: 'single-experiment dataset',
            experiment_name: 'qa-smiles-298',
            adsorbent: 'Activated carbon',
            adsorbate: 'CO2',
            temperature_k: 298.15,
            pressure_basis: 'absolute',
            pressure_unit: 'Pa',
            uptake_unit: 'mol/kg',
            observation_count: 4,
            best_model: 'langmuir',
            results: [
                {
                    model: 'langmuir',
                    name: 'Langmuir',
                    status: 'success',
                    convergence_message: 'ok',
                    function_evaluations: 5,
                    jacobian_rank: 2,
                    condition_number: null,
                    parameters: [],
                    metrics: {
                        sse: 1,
                        rmse: 0.5,
                        mae: 0.4,
                        r_squared: 0.98,
                        adjusted_r_squared: 0.96,
                        chi_square: null,
                        aic: 2,
                        aicc: 8,
                        bic: 3,
                    },
                    observed_predictions: [],
                    curve: [],
                    warnings: [],
                    rank: 1,
                },
            ],
            summary: 'Langmuir selected.',
        };
        TestBed.inject(CoreWorkspaceStore).fittingResult.set(fittingResult);
        fixture.detectChanges();

        const root = fixture.nativeElement as HTMLElement;
        expect(root.querySelector('.fitting-result-panel')?.textContent).toContain('Best model');
        expect(root.querySelector('.fitting-result-panel')?.textContent).toContain('Langmuir');
        expect(root.querySelector('.fitting-result-panel')?.textContent).toContain('0.5');
        expect(root.querySelector('.fitting-result-row-best')).not.toBeNull();
    });

    it('sends model parameter bounds and exposes cancellation for a running fit', async () => {
        let cancellationRequested = false;
        let submittedPayload: Record<string, unknown> | null = null;
        fetchMock.mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
            const url = String(input);
            if (url.endsWith('/system/configuration')) {
                return {
                    ok: true,
                    status: 200,
                    json: async () => ({
                        status: 'success',
                        supported_optimizers: ['trf', 'dogbox'],
                        default_optimizer: 'trf',
                        default_max_evaluations: 100,
                        max_evaluations_bounds: { minimum: 10, maximum: 1000 },
                        weighting_options: ['unweighted', 'inverse_sigma'],
                        default_weighting: 'unweighted',
                        display_units: {
                            pressure: ['bar'],
                            uptake: ['mmol/g'],
                            default_pressure: 'bar',
                            default_uptake: 'mmol/g',
                        },
                        parameter_defaults: { lower: 0, upper: 100, initial: 1 },
                    }),
                };
            }
            if (url.includes('/fitting/models')) {
                return {
                    ok: true,
                    status: 200,
                    json: async () => ({
                        status: 'success',
                        pressure_unit: 'bar',
                        uptake_unit: 'mmol/g',
                        models: [{
                            key: 'langmuir',
                            name: 'Langmuir',
                            equation_latex: 'q = q_{sat}kp/(1+kp)',
                            assumptions: 'Single-site adsorption',
                            pressure_requirement: 'Positive pressure',
                            requires_temperature: false,
                            reference: 'Langmuir (1918)',
                            parameters: [
                                { name: 'k', label: 'Affinity', lower: 0.01, upper: 10, initial: 1, unit: 'bar^-1' },
                                { name: 'qsat', label: 'Capacity', lower: 0.01, upper: 100, initial: 20, unit: 'mmol/g' },
                            ],
                        }],
                    }),
                };
            }
            if (url.endsWith('/fitting/run')) {
                submittedPayload = JSON.parse(String(init?.body)) as Record<string, unknown>;
                return {
                    ok: true,
                    status: 200,
                    json: async () => ({ job_id: 'fitting-cancel-test', poll_interval: 0.01 }),
                };
            }
            if (url.endsWith('/fitting/jobs/fitting-cancel-test') && init?.method === 'DELETE') {
                cancellationRequested = true;
                return {
                    ok: true,
                    status: 200,
                    json: async () => ({ status: 'cancelled', job_id: 'fitting-cancel-test' }),
                };
            }
            if (url.endsWith('/fitting/jobs/fitting-cancel-test')) {
                return {
                    ok: true,
                    status: 200,
                    json: async () => ({
                        job_id: 'fitting-cancel-test',
                        job_type: 'fitting',
                        status: cancellationRequested ? 'cancelled' : 'running',
                        poll_interval: 0.01,
                        result: null,
                        error: null,
                    }),
                };
            }
            if (url.endsWith('/datasets')) {
                return {
                    ok: true,
                    status: 200,
                    json: async () => ({
                        datasets: [{
                            id: 1,
                            name: 'single-experiment dataset',
                            source: 'uploaded',
                            created_at: '2026-08-13T00:00:00Z',
                            experiment_count: 1,
                            observation_count: 4,
                            tags: [],
                            description: '',
                        }],
                    }),
                };
            }
            if (url.includes('/datasets/1/experiments')) {
                return {
                    ok: true,
                    status: 200,
                    json: async () => ({
                        experiments: [{
                            id: 42,
                            dataset_id: 1,
                            external_key: 'qa-smiles-298',
                            name: 'qa-smiles-298',
                            adsorbent: 'Activated carbon',
                            adsorbates: ['CO2'],
                            temperature_k: 298.15,
                            pressure_basis: 'absolute',
                            observation_count: 4,
                            fitting_eligible: true,
                            ineligibility_reason: null,
                        }],
                    }),
                };
            }
            throw new Error(`Unhandled URL ${url}`);
        });

        const fixture = TestBed.createComponent(ModelsPageComponent);
        fixture.detectChanges();
        await fixture.whenStable();
        fixture.detectChanges();

        const root = fixture.nativeElement as HTMLElement;
        const datasetSelect = root.querySelector<HTMLSelectElement>('#fitting-dataset-control');
        datasetSelect!.value = '1';
        datasetSelect!.dispatchEvent(new Event('change'));
        await fixture.whenStable();
        fixture.detectChanges();

        const store = TestBed.inject(CoreWorkspaceStore);
        store.setModelParameters('langmuir', {
            k: { min: 1.5, max: 3 },
            qsat: { min: 10, max: 30 },
        });
        root.querySelector<HTMLButtonElement>('.fitting-action-primary')!.click();

        await vi.waitFor(() => expect(store.fittingJobId()).toBe('fitting-cancel-test'));
        fixture.detectChanges();
        const cancelButton = Array.from(root.querySelectorAll<HTMLButtonElement>('button'))
            .find((button) => button.textContent?.includes('Cancel Fitting'));
        expect(cancelButton).not.toBeUndefined();
        cancelButton!.click();

        await vi.waitFor(() => expect(cancellationRequested).toBe(true));
        await vi.waitFor(() => expect(store.fittingRunning()).toBe(false));
        fixture.detectChanges();

        const submittedConfiguration = submittedPayload?.['parameter_configuration'] as unknown as Record<string, Record<string, { lower: number; upper: number; initial: number }>>;
        expect(submittedConfiguration['langmuir']['k']).toEqual({ lower: 1.5, upper: 3, initial: 1.5 });
        expect(submittedConfiguration['langmuir']['qsat']).toEqual({ lower: 10, upper: 30, initial: 20 });
        expect(store.fittingStatus()).toBe('[INFO] Job was cancelled.');
        expect(store.fittingJobId()).toBeNull();
    });
});
