// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, effect, inject, input, output, signal, untracked } from '@angular/core';
import { CoreWorkspaceStore } from '../../core/state/core-workspace.store';
import type { DatasetSummary } from '../../models/dataset.model';
import { fetchDeleteImpact } from '../../services/dataset.service';
import { ConfirmDialogComponent } from '../../shared/components/confirm-dialog/confirm-dialog.component';

/** Confirm and perform a dataset deletion, stating exactly what is removed. */
@Component({
    selector: 'adsmod-dataset-delete-dialog',
    standalone: true,
    imports: [ConfirmDialogComponent],
    template: `
        <adsmod-confirm-dialog
            [open]="dataset() !== null"
            [title]="'Delete ' + (dataset()?.name ?? 'dataset')"
            [message]="message()"
            confirmLabel="Delete dataset"
            (confirmed)="confirm()"
            (closed)="closed.emit()"
        />
    `,
})
export class DatasetDeleteDialogComponent {
    readonly dataset = input<DatasetSummary | null>(null);
    readonly closed = output<void>();
    readonly deleted = output<number>();

    private readonly store = inject(CoreWorkspaceStore);
    protected readonly message = signal('');

    constructor() {
        effect(() => {
            const dataset = this.dataset();
            if (dataset) untracked(() => void this.describe(dataset));
        });
    }

    private async describe(dataset: DatasetSummary): Promise<void> {
        this.message.set(
            `This permanently removes ${dataset.experiment_count} experiments and ${dataset.observation_count} measurements. Materials and species are kept.`,
        );
        const result = await fetchDeleteImpact(dataset.id);
        if (result.data && this.dataset()?.id === dataset.id) {
            const runs = result.data.fitting_run_count;
            const parts = [`${result.data.experiment_count} experiments`, `${result.data.observation_count} measurements`];
            if (runs) parts.push(`${runs} saved fitting run${runs === 1 ? '' : 's'}`);
            const removed = `${parts.slice(0, -1).join(', ')} and ${parts[parts.length - 1]}`;
            this.message.set(`This permanently removes ${removed}. Materials and species stay available for other datasets.`);
        }
    }

    protected async confirm(): Promise<void> {
        const dataset = this.dataset();
        if (!dataset) return;
        const removed = await this.store.deleteDataset(dataset.id);
        this.closed.emit();
        if (removed) this.deleted.emit(dataset.id);
    }
}
