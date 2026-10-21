import type { CanDeactivateFn } from '@angular/router';
import type { SettingsPageComponent } from './settings-page.component';

export const settingsAutosaveGuard: CanDeactivateFn<SettingsPageComponent> = (component) =>
    component.flushPendingChanges();
