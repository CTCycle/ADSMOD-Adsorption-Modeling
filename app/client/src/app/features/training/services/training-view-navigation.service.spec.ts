import { TestBed } from '@angular/core/testing';
import { Router } from '@angular/router';
import { describe, expect, it, vi } from 'vitest';
import { TrainingViewNavigationService } from './training-view-navigation.service';

describe('TrainingViewNavigationService', () => {
    it('navigates to the requested training view route', async () => {
        const router = {
            navigate: vi.fn().mockResolvedValue(true),
        };

        TestBed.configureTestingModule({
            providers: [
                TrainingViewNavigationService,
                { provide: Router, useValue: router },
            ],
        });

        const service = TestBed.inject(TrainingViewNavigationService);

        await expect(service.navigateTo('dashboard')).resolves.toBe(true);
        expect(router.navigate).toHaveBeenCalledWith(['/training', 'dashboard']);
    });
});