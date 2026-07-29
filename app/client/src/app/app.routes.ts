import { Routes } from '@angular/router';
import { CoreShellComponent } from './layout/core-shell.component';

export const routes: Routes = [
    {
        path: '',
        component: CoreShellComponent,
        children: [
            { path: '', pathMatch: 'full', redirectTo: 'datasets' },
            { path: 'datasets', loadComponent: () => import('./features/datasets/datasets-page.component').then((m) => m.DatasetsPageComponent) },
            { path: 'dashboards', loadComponent: () => import('./features/dashboards/dashboards-page.component').then((m) => m.DashboardsPageComponent) },
            { path: 'fitting', loadComponent: () => import('./features/fitting/models-page.component').then((m) => m.ModelsPageComponent) },
            { path: 'training', pathMatch: 'full', redirectTo: 'training/processing' },
            {
                path: 'training/:view',
                loadComponent: () =>
                    import('./features/training/pages/machine-learning-page.component').then((m) => m.MachineLearningPageComponent),
            },
        ],
    },
    { path: '**', redirectTo: 'datasets' },
];
