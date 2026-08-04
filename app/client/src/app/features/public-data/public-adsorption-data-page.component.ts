import { Component, signal } from '@angular/core';
import { NistCollectionRowsComponent } from '../nist/nist-collection-rows.component';
import type { NISTCategoryKey } from '../../models/nist.model';

@Component({
    selector: 'adsmod-public-adsorption-data-page',
    standalone: true,
    imports: [NistCollectionRowsComponent],
    template: `
        <div class="data-page public-source-page">
            <section class="console-card public-source-card">
                <div class="card-title-row public-source-heading">
                    <div>
                        <p class="eyebrow">Public adsorption data</p>
                        <h2>NIST-A Collection</h2>
                        <p>Download public adsorption experiments into the local canonical workspace for analysis, fitting, and training.</p>
                    </div>
                    <span class="source-badge">NIST ISODB</span>
                </div>
                <adsmod-nist-collection-rows [categories]="categories" (statusUpdate)="appendStatus($event)" />
            </section>
            <section class="console-card nist-log-card public-source-log">
                <div class="card-title-row">
                    <div>
                        <p class="eyebrow">Activity</p>
                        <h2>NIST-A Status Updates</h2>
                    </div>
                    <button class="button quiet nist-clear-button" type="button" aria-label="Clear status updates" title="Clear status updates" (click)="clearStatus()">
                        <svg aria-hidden="true" viewBox="0 0 24 24"><path d="m3 17 8-11a2 2 0 0 1 3 0l4 4a2 2 0 0 1 0 3l-7 7H5a2 2 0 0 1-2-2v-1Z" /><path d="m15 21 6-6" /></svg>
                    </button>
                </div>
                @if (status().length) {
                    <pre class="reference-log" aria-live="polite">{{ status().join('\n\n') }}</pre>
                } @else {
                    <p class="nist-log-empty" aria-live="polite">NIST-A experiment updates will appear here.</p>
                }
            </section>
        </div>
    `,
})
export class PublicAdsorptionDataPageComponent {
    protected readonly categories: readonly NISTCategoryKey[] = ['experiments'];
    protected readonly status = signal<string[]>([]);

    protected appendStatus(message: string): void {
        this.status.update((entries) => [...entries, message]);
    }

    protected clearStatus(): void {
        this.status.set([]);
    }
}
