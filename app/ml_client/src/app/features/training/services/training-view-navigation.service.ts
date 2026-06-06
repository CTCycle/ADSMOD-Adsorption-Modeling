import { Injectable, inject } from '@angular/core';
import { Router } from '@angular/router';
import type { TrainingViewId } from '../../../core/state/training-workspace.store';

@Injectable({ providedIn: 'root' })
export class TrainingViewNavigationService {
    private readonly router = inject(Router);

    navigateTo(view: TrainingViewId): Promise<boolean> {
        return this.router.navigate(['/training', view]);
    }
}