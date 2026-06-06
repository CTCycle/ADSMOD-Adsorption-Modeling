import { Routes } from '@angular/router';
import { MlShellComponent } from './layout/ml-shell.component';

export const routes: Routes = [
    {
        path: '',
        component: MlShellComponent,
        children: [
            { path: '', pathMatch: 'full', redirectTo: 'training/processing' },
            { path: 'training', pathMatch: 'full', redirectTo: 'training/processing' },
            {
                path: 'training/:view',
                loadComponent: () =>
                    import('./features/training/pages/machine-learning-page.component').then((m) => m.MachineLearningPageComponent),
            },
        ],
    },
    { path: '**', redirectTo: 'training/processing' },
];
