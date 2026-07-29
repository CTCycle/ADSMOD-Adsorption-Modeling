import { Component, signal } from '@angular/core';
import { NistCollectionRowsComponent } from './nist-collection-rows.component';
@Component({selector:'adsmod-nist-data-page',standalone:true,imports:[NistCollectionRowsComponent],template:`<div class="nist-data-page"><section class="console-card nist-fetch-card"><div class="card-title-row"><h2>NIST-A Collection</h2></div><adsmod-nist-collection-rows (statusUpdate)="appendStatus($event)" /></section><section class="console-card nist-log-card"><div class="card-title-row"><h2>NIST-A Status Updates</h2><button class="button quiet nist-clear-button" type="button" aria-label="Clear status updates" title="Clear status updates" (click)="clearStatus()"><svg aria-hidden="true" viewBox="0 0 24 24"><path d="m3 17 8-11a2 2 0 0 1 3 0l4 4a2 2 0 0 1 0 3l-7 7H5a2 2 0 0 1-2-2v-1Z" /><path d="m15 21 6-6" /></svg></button></div>@if (status().length) { <pre class="reference-log" aria-live="polite">{{ status().join('\n\n') }}</pre> } @else { <p class="nist-log-empty" aria-live="polite">NIST-A updates will appear here.</p> }</section></div>`})
export class NistDataPageComponent {
    protected readonly status = signal<string[]>([]);

    protected appendStatus(message: string): void {
        this.status.update((entries) => [...entries, message]);
    }

    protected clearStatus(): void {
        this.status.set([]);
    }
}
