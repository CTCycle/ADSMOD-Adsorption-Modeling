// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';

@Component({
    selector: 'adsmod-root',
    standalone: true,
    imports: [RouterOutlet],
    template: '<router-outlet />',
})
export class AppComponent {}
