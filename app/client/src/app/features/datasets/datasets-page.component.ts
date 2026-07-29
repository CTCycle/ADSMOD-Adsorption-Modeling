import { Component } from '@angular/core';
import { NistDataPageComponent } from '../nist/nist-data-page.component';
import { SourcePageComponent } from '../source/source-page.component';

@Component({ selector: 'adsmod-datasets-page', standalone: true, imports: [SourcePageComponent, NistDataPageComponent], template: `<adsmod-source-page /><adsmod-nist-data-page />` })
export class DatasetsPageComponent {}
