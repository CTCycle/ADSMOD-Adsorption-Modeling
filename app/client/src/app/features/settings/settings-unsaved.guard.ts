import type { CanDeactivateFn } from '@angular/router';
import type { SettingsPageComponent } from './settings-page.component';

export const settingsUnsavedGuard: CanDeactivateFn<SettingsPageComponent> = (component) =>
    component.canLeave();
