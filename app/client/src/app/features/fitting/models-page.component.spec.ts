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
});
