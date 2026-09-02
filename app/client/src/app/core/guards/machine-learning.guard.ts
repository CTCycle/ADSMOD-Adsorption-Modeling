import { inject } from '@angular/core';
import { CanActivateFn, Router } from '@angular/router';
import { machineLearningAvailable } from '../../services/system.service';

export const machineLearningGuard: CanActivateFn = async () => {
    const router = inject(Router);
    return await machineLearningAvailable() ? true : router.createUrlTree(['/datasets']);
};
