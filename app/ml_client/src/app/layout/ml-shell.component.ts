import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';

@Component({
    selector: 'adsmod-ml-shell',
    standalone: true,
    imports: [RouterOutlet],
    template: '<router-outlet />',
})
export class MlShellComponent {}
