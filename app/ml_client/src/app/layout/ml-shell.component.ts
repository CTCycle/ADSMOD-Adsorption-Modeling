import { Component } from '@angular/core';
import { RouterOutlet } from '@angular/router';

@Component({
    selector: 'adsmod-ml-shell',
    standalone: true,
    imports: [RouterOutlet],
    template: `
        <div class="app-container">
            <main class="app-main">
                <router-outlet />
            </main>
        </div>
    `,
})
export class MlShellComponent {}
