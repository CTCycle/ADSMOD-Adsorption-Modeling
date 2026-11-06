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
            { path: '', pathMatch: 'full', redirectTo: 'home' },
            { path: 'home', loadComponent: () => import('./features/home/home-page.component').then((m) => m.HomePageComponent) },
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
            // "/docs" is reserved by the backend (OpenAPI UI), so the in-app docs live at "/help".
            { path: 'help', loadComponent: () => import('./features/docs/docs-page.component').then((m) => m.DocsPageComponent) },
            { path: 'settings', canDeactivate: [settingsAutosaveGuard], loadComponent: () => import('./features/settings/settings-page.component').then((m) => m.SettingsPageComponent) },
            {
                path: 'modeling',
                loadComponent: () => import('./features/modeling/modeling-page.component').then((m) => m.ModelingPageComponent),
                children: [
                    { path: '', pathMatch: 'full', redirectTo: 'fitting' },
                    { path: 'fitting', loadComponent: () => import('./features/modeling/fitting/sessions-view.component').then((m) => m.SessionsViewComponent) },
                    { path: 'fitting/new', loadComponent: () => import('./features/modeling/fitting/session-wizard.component').then((m) => m.SessionWizardComponent) },
                    { path: 'fitting/:sessionId', loadComponent: () => import('./features/modeling/fitting/session-view.component').then((m) => m.SessionViewComponent) },
                    { path: 'models', loadComponent: () => import('./features/modeling/models/adsorption-models-view.component').then((m) => m.AdsorptionModelsViewComponent) },
                    {
                        path: 'deep-learning',
                        pathMatch: 'full',
                        canActivate: [machineLearningEntryGuard],
                        loadComponent: () =>
                            import('./features/training/pages/machine-learning-page.component').then((m) => m.MachineLearningPageComponent),
                    },
                    {
                        path: 'deep-learning/:view',
                        canActivate: [machineLearningGuard],
                        loadComponent: () =>
                            import('./features/training/pages/machine-learning-page.component').then((m) => m.MachineLearningPageComponent),
                    },
                ],
            },
        ],
    },
    { path: '**', redirectTo: 'home' },
];
