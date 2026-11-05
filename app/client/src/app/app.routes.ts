// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Routes } from '@angular/router';
import { machineLearningEntryGuard, machineLearningGuard } from './core/guards/machine-learning.guard';
import { CoreShellComponent } from './layout/core-shell.component';
import { settingsAutosaveGuard } from './features/settings/settings-autosave.guard';
import { unsavedChangesGuard } from './core/guards/unsaved-changes.guard';

export const routes: Routes = [
    {
        path: '',
        component: CoreShellComponent,
        children: [
            { path: '', pathMatch: 'full', redirectTo: 'datasets' },
            { path: 'datasets', loadComponent: () => import('./features/datasets/datasets-page.component').then((m) => m.DatasetsPageComponent) },
            { path: 'datasets/:id', canDeactivate: [unsavedChangesGuard], loadComponent: () => import('./features/datasets/dataset-details-page.component').then((m) => m.DatasetDetailsPageComponent) },
            { path: 'sources', pathMatch: 'full', redirectTo: 'sources/catalog' },
            { path: 'sources/:view', loadComponent: () => import('./features/sources/sources-page.component').then((m) => m.SourcesPageComponent) },
            {
                path: 'dashboards',
                loadComponent: () => import('./features/dashboards/dashboards-page.component').then((m) => m.DashboardsPageComponent),
                children: [
                    { path: '', pathMatch: 'full', redirectTo: 'overview' },
                    { path: 'overview', loadComponent: () => import('./features/dashboards/overview/overview-view.component').then((m) => m.OverviewViewComponent) },
                    { path: 'explore', loadComponent: () => import('./features/dashboards/explore/explore-view.component').then((m) => m.ExploreViewComponent) },
                ],
            },
            { path: 'fitting', loadComponent: () => import('./features/fitting/models-page.component').then((m) => m.ModelsPageComponent) },
            // "/docs" is reserved by the backend (OpenAPI UI), so the in-app docs live at "/help".
            { path: 'help', loadComponent: () => import('./features/docs/docs-page.component').then((m) => m.DocsPageComponent) },
            { path: 'settings', canDeactivate: [settingsAutosaveGuard], loadComponent: () => import('./features/settings/settings-page.component').then((m) => m.SettingsPageComponent) },
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
