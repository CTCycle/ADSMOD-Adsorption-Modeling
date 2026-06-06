import { expect, test } from '@playwright/test';

type MlStatusScenario = 'idle' | 'training';

const prepareVisualPage = async (page: import('@playwright/test').Page) => {
    await page.addInitScript(() => {
        window.localStorage.clear();
    });
};

const disableMotion = async (page: import('@playwright/test').Page) => {
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.addStyleTag({
        content: `
            *,
            *::before,
            *::after {
                animation-duration: 0s !important;
                animation-delay: 0s !important;
                transition-duration: 0s !important;
                transition-delay: 0s !important;
                caret-color: transparent !important;
            }
        `,
    });
};

const statusBodyFor = (scenario: MlStatusScenario) => scenario === 'training'
    ? {
        is_training: true,
        current_epoch: 8,
        total_epochs: 12,
        progress: 66.7,
        poll_interval: 2,
        metrics: {
            loss: 0.1842,
            val_loss: 0.2216,
            masked_r2: 0.9321,
            val_masked_r2: 0.9148,
        },
        history: [
            { epoch: 1, loss: 0.8611, val_loss: 0.9014, masked_r2: 0.412, val_masked_r2: 0.366 },
            { epoch: 2, loss: 0.6449, val_loss: 0.7022, masked_r2: 0.584, val_masked_r2: 0.531 },
            { epoch: 3, loss: 0.5211, val_loss: 0.5758, masked_r2: 0.681, val_masked_r2: 0.639 },
            { epoch: 4, loss: 0.4032, val_loss: 0.4625, masked_r2: 0.774, val_masked_r2: 0.726 },
            { epoch: 5, loss: 0.3118, val_loss: 0.3554, masked_r2: 0.846, val_masked_r2: 0.801 },
            { epoch: 6, loss: 0.2541, val_loss: 0.2988, masked_r2: 0.889, val_masked_r2: 0.857 },
            { epoch: 7, loss: 0.2193, val_loss: 0.2517, masked_r2: 0.911, val_masked_r2: 0.889 },
            { epoch: 8, loss: 0.1842, val_loss: 0.2216, masked_r2: 0.9321, val_masked_r2: 0.9148 },
        ],
        log: [
            'Dataset loaded: zeolite_curated_v2',
            'Epoch 8/12 complete',
            'Validation improved to 0.9148 masked R2',
        ],
    }
    : {
        is_training: false,
        current_epoch: 0,
        total_epochs: 12,
        progress: 0,
        poll_interval: 2,
        metrics: {},
        history: [],
        log: ['Ready to start training...'],
    };

const mockMlApi = async (page: import('@playwright/test').Page, scenario: MlStatusScenario) => {
    await page.route('**/api/**', async (route) => {
        const request = route.request();
        const url = new URL(request.url());

        if (request.method() === 'GET' && url.pathname === '/api/training/processed-datasets') {
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({
                    datasets: [
                        {
                            dataset_label: 'zeolite_curated_v2',
                            dataset_hash: 'hash-zeolite-curated-v2',
                            train_samples: 1960,
                            validation_samples: 420,
                        },
                        {
                            dataset_label: 'nist_screening_fraction_035',
                            dataset_hash: 'hash-nist-screening-035',
                            train_samples: 1420,
                            validation_samples: 304,
                        },
                    ],
                }),
            });
            return;
        }

        if (request.method() === 'GET' && url.pathname === '/api/training/checkpoints') {
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({
                    checkpoints: [
                        { name: 'scads_zeolite_ep08', epochs_trained: 8, final_loss: 0.1842, is_compatible: true },
                        { name: 'scads_nist_ep12', epochs_trained: 12, final_loss: 0.2216, is_compatible: true },
                    ],
                }),
            });
            return;
        }

        if (request.method() === 'GET' && url.pathname === '/api/training/checkpoints/scads_zeolite_ep08') {
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({
                    name: 'scads_zeolite_ep08',
                    epochs_trained: 8,
                    final_loss: 0.1842,
                    final_accuracy: 0.9182,
                    is_compatible: true,
                    created_at: '2026-05-31T13:20:00Z',
                    configuration: {
                        batch_size: 16,
                        shuffle_dataset: true,
                        max_buffer_size: 256,
                        selected_model: 'SCADS Series',
                        dropout_rate: 0.1,
                        num_attention_heads: 2,
                        num_encoders: 2,
                        molecular_embedding_size: 64,
                        epochs: 12,
                        dataloader_workers: 0,
                        prefetch_factor: 1,
                        pin_memory: true,
                        use_device_GPU: true,
                        device_ID: 0,
                        use_mixed_precision: false,
                        use_jit: false,
                        jit_backend: 'inductor',
                        use_lr_scheduler: false,
                        initial_lr: 0.0001,
                        target_lr: 0.00001,
                        constant_steps: 5,
                        decay_steps: 10,
                        save_checkpoints: false,
                        checkpoints_frequency: 5,
                    },
                    metadata: {
                        available: true,
                        dataset_label: 'zeolite_curated_v2',
                        total_samples: 2380,
                        train_samples: 1960,
                        validation_samples: 420,
                    },
                    history: null,
                }),
            });
            return;
        }

        if (request.method() === 'GET' && url.pathname === '/api/training/status') {
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify(statusBodyFor(scenario)),
            });
            return;
        }

        if (request.method() === 'GET' && url.pathname === '/api/training/dataset-info') {
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({
                    available: true,
                    dataset_label: 'zeolite_curated_v2',
                    created_at: '2026-05-31T13:20:00Z',
                    total_samples: 2380,
                    train_samples: 1960,
                    validation_samples: 420,
                    sample_size: 0.4,
                    validation_size: 0.1765,
                    min_measurements: 6,
                    max_measurements: 42,
                    smile_sequence_size: 96,
                    max_pressure: 50,
                    max_uptake: 22.4,
                    smile_vocabulary_size: 128,
                    adsorbent_vocabulary_size: 44,
                    normalization_stats: {
                        pressure_mean: 13.2,
                        uptake_mean: 4.9,
                    },
                }),
            });
            return;
        }

        if (request.method() === 'GET' && url.pathname === '/api/training/dataset-sources') {
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({
                    datasets: [
                        {
                            source: 'uploaded',
                            dataset_name: 'zeolite_batch_august',
                            display_name: 'zeolite_batch_august',
                            row_count: 1284,
                        },
                        {
                            source: 'nist',
                            dataset_name: 'nist_screening_fraction_035',
                            display_name: 'nist_screening_fraction_035',
                            row_count: 1724,
                        },
                    ],
                }),
            });
            return;
        }

        await route.fulfill({
            status: 404,
            contentType: 'application/json',
            body: JSON.stringify({
                detail: `Unhandled visual mock for ${request.method()} ${url.pathname}`,
            }),
        });
    });
};

test.describe('ml visual regression', () => {
    test('processing page remains visually stable', async ({ page }) => {
        await mockMlApi(page, 'idle');
        await prepareVisualPage(page);
        await page.goto('/training/processing');
        await disableMotion(page);
        await expect(page.getByRole('heading', { name: 'Data Processing' })).toBeVisible();
        await expect(page).toHaveScreenshot('ml-processing-page.png', { fullPage: true });
    });

    test('datasets page remains visually stable', async ({ page }) => {
        await mockMlApi(page, 'idle');
        await prepareVisualPage(page);
        await page.goto('/training/datasets');
        await disableMotion(page);
        await expect(page.getByRole('heading', { name: 'Train datasets' })).toBeVisible();
        await expect(page).toHaveScreenshot('ml-datasets-page.png', { fullPage: true });
    });

    test('checkpoints page remains visually stable', async ({ page }) => {
        await mockMlApi(page, 'idle');
        await prepareVisualPage(page);
        await page.goto('/training/checkpoints');
        await disableMotion(page);
        await expect(page.getByRole('heading', { name: 'Checkpoints' })).toBeVisible();
        await expect(page).toHaveScreenshot('ml-checkpoints-page.png', { fullPage: true });
    });

    test('new training wizard page 1 remains visually stable', async ({ page }) => {
        await mockMlApi(page, 'idle');
        await prepareVisualPage(page);
        await page.goto('/training/datasets');
        await disableMotion(page);
        await page.locator('.split-table-row').first().click();
        await page.getByRole('button', { name: 'Open Training Setup' }).click();
        await expect(page.getByRole('heading', { name: 'New Training Wizard' })).toBeVisible();
        await expect(page).toHaveScreenshot('ml-new-training-wizard-page-1.png', { fullPage: true });
    });

    test('new training wizard final page remains visually stable', async ({ page }) => {
        await mockMlApi(page, 'idle');
        await prepareVisualPage(page);
        await page.goto('/training/datasets');
        await disableMotion(page);
        await page.locator('.split-table-row').first().click();
        await page.getByRole('button', { name: 'Open Training Setup' }).click();
        for (let index = 0; index < 4; index += 1) {
            await page.getByRole('button', { name: 'Next' }).click();
        }
        await expect(page.getByText('Training Name')).toBeVisible();
        await expect(page).toHaveScreenshot('ml-new-training-wizard-page-5.png', { fullPage: true });
    });

    test('resume training wizard remains visually stable', async ({ page }) => {
        await mockMlApi(page, 'idle');
        await prepareVisualPage(page);
        await page.goto('/training/checkpoints');
        await disableMotion(page);
        await page.locator('.split-table-row').first().click();
        await page.getByRole('button', { name: 'Resume Training' }).click();
        await expect(page.getByRole('heading', { name: 'Resume Training Wizard' })).toBeVisible();
        await expect(page).toHaveScreenshot('ml-resume-training-wizard-page.png', { fullPage: true });
    });

    test('dataset info modal remains visually stable', async ({ page }) => {
        await mockMlApi(page, 'idle');
        await prepareVisualPage(page);
        await page.goto('/training/datasets');
        await disableMotion(page);
        await page.locator('.icon-action-button[title="View Metadata"]').first().click();
        await expect(page.getByRole('heading', { name: 'Dataset Metadata' })).toBeVisible();
        await expect(page).toHaveScreenshot('ml-dataset-info-modal-page.png', { fullPage: true });
    });

    test('dashboard idle chart state remains visually stable', async ({ page }) => {
        await mockMlApi(page, 'idle');
        await prepareVisualPage(page);
        await page.goto('/training/dashboard');
        await disableMotion(page);
        await expect(page.getByRole('heading', { name: 'Training Dashboard' })).toBeVisible();
        await expect(page).toHaveScreenshot('ml-dashboard-idle-page.png', { fullPage: true });
    });

    test('dashboard populated chart state remains visually stable', async ({ page }) => {
        await mockMlApi(page, 'training');
        await prepareVisualPage(page);
        await page.goto('/training/dashboard');
        await disableMotion(page);
        await expect(page.getByRole('heading', { name: 'Training Dashboard' })).toBeVisible();
        await expect(page).toHaveScreenshot('ml-dashboard-populated-page.png', { fullPage: true });
    });
});
