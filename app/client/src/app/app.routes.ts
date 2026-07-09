import { Routes } from '@angular/router';
import { CoreShellComponent } from './layout/core-shell.component';

export const routes: Routes = [
    {
        path: '',
        component: CoreShellComponent,
        children: [
            { path: '', pathMatch: 'full', redirectTo: 'source' },
            { path: 'source', loadComponent: () => import('./features/source/source-page.component').then((m) => m.SourcePageComponent) },
            { path: 'fitting', loadComponent: () => import('./features/fitting/models-page.component').then((m) => m.ModelsPageComponent) },
            { path: 'training', pathMatch: 'full', redirectTo: 'training/processing' },
            {
                path: 'training/:view',
                loadComponent: () =>
                    import('./features/training/pages/machine-learning-page.component').then((m) => m.MachineLearningPageComponent),
            },
        ],
    },
    { path: '**', redirectTo: 'source' },
];
