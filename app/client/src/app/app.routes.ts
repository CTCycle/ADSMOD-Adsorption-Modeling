import { Routes } from '@angular/router';
import { machineLearningEntryGuard, machineLearningGuard } from './core/guards/machine-learning.guard';
import { CoreShellComponent } from './layout/core-shell.component';

export const routes: Routes = [
    {
        path: '',
        component: CoreShellComponent,
        children: [
            { path: '', pathMatch: 'full', redirectTo: 'datasets' },
            { path: 'datasets', loadComponent: () => import('./features/datasets/custom-datasets-page.component').then((m) => m.CustomDatasetsPageComponent) },
            { path: 'public-data', loadComponent: () => import('./features/public-data/public-adsorption-data-page.component').then((m) => m.PublicAdsorptionDataPageComponent) },
            { path: 'public-materials', loadComponent: () => import('./features/public-materials/public-materials-page.component').then((m) => m.PublicMaterialsPageComponent) },
            { path: 'dashboards', loadComponent: () => import('./features/dashboards/dashboards-page.component').then((m) => m.DashboardsPageComponent) },
            { path: 'fitting', loadComponent: () => import('./features/fitting/models-page.component').then((m) => m.ModelsPageComponent) },
            {
                path: 'training',
                pathMatch: 'full',
                canActivate: [machineLearningEntryGuard],
                loadComponent: () =>
                    import('./features/training/pages/machine-learning-page.component').then((m) => m.MachineLearningPageComponent),
            },
            {
                path: 'training/:view',
                canActivate: [machineLearningGuard],
                loadComponent: () =>
                    import('./features/training/pages/machine-learning-page.component').then((m) => m.MachineLearningPageComponent),
            },
        ],
    },
    { path: '**', redirectTo: 'datasets' },
];
