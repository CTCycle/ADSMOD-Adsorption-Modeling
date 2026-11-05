// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import type { CanDeactivateFn } from '@angular/router';

export interface HasUnsavedChanges {
    hasUnsavedChanges(): boolean;
}

export const UNSAVED_CHANGES_MESSAGE = 'You have unsaved changes. Leave this page and discard them?';

/** Ask before navigating away from a page that holds unsaved edits. */
export const unsavedChangesGuard: CanDeactivateFn<HasUnsavedChanges> = (component) =>
    !component.hasUnsavedChanges() || window.confirm(UNSAVED_CHANGES_MESSAGE);
