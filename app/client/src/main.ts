// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { bootstrapApplication } from '@angular/platform-browser';
import { provideRouter, withComponentInputBinding } from '@angular/router';
import { provideHttpClient } from '@angular/common/http';
import { AppComponent } from './app/app.component';
import { routes } from './app/app.routes';

bootstrapApplication(AppComponent, {
    providers: [
        provideHttpClient(),
        provideRouter(routes, withComponentInputBinding()),
    ],
}).catch((error: unknown) => {
    console.error(error);
});
