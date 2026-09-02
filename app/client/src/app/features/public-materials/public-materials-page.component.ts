import { Component, signal } from '@angular/core';
import { NistCollectionRowsComponent } from '../nist/nist-collection-rows.component';
import type { NISTCategoryKey } from '../../models/nist.model';

@Component({
    selector: 'adsmod-public-materials-page',
    standalone: true,
    imports: [NistCollectionRowsComponent],
    template: `
        <div class="data-page public-source-page public-materials-page">
            <div class="page-intro-card console-card">
                <p class="eyebrow">Public reference information</p>
                <h2>Materials &amp; Adsorbates</h2>
                <p>Retrieve public identities and properties for adsorbent materials and adsorbate species. NIST provides the records; PubChem is used only for the existing enrichment action.</p>
            </div>
            <section class="console-card public-category-card">
                <div class="card-title-row">
                    <div><p class="eyebrow">Guest species</p><h2>Adsorbates</h2><p>Gas or liquid species referenced by public adsorption records.</p></div>
                </div>
                <adsmod-nist-collection-rows [categories]="guestCategories" (statusUpdate)="appendStatus($event)" />
            </section>
            <section class="console-card public-category-card">
                <div class="card-title-row">
                    <div><p class="eyebrow">Host materials</p><h2>Adsorbent Materials</h2><p>Public material identities and registry-linked properties.</p></div>
                </div>
                <adsmod-nist-collection-rows [categories]="hostCategories" (statusUpdate)="appendStatus($event)" />
            </section>
            <section class="console-card nist-log-card public-source-log">
                <div class="card-title-row">
                    <div><p class="eyebrow">Activity</p><h2>Reference Data Status</h2></div>
                    <button class="button quiet nist-clear-button" type="button" aria-label="Clear status updates" title="Clear status updates" (click)="clearStatus()">
                        <svg aria-hidden="true" viewBox="0 0 24 24"><path d="m3 17 8-11a2 2 0 0 1 3 0l4 4a2 2 0 0 1 0 3l-7 7H5a2 2 0 0 1-2-2v-1Z" /><path d="m15 21 6-6" /></svg>
                    </button>
                </div>
                @if (status().length) {
                    <pre class="reference-log" aria-live="polite">{{ status().join('\n\n') }}</pre>
                } @else {
                    <p class="nist-log-empty" aria-live="polite">Reference data updates will appear here.</p>
                }
            </section>
        </div>
    `,
})
export class PublicMaterialsPageComponent {
    protected readonly guestCategories: readonly NISTCategoryKey[] = ['guest'];
    protected readonly hostCategories: readonly NISTCategoryKey[] = ['host'];
    protected readonly status = signal<string[]>([]);

    protected appendStatus(message: string): void {
        this.status.update((entries) => [...entries, message]);
    }

    protected clearStatus(): void {
        this.status.set([]);
    }
}
