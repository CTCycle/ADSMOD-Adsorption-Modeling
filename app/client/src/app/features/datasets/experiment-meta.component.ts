// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { DatePipe } from '@angular/common';
import { Component, computed, input, signal } from '@angular/core';
import type { ExperimentComponent, ExperimentSummary, PressureBasis, UnitCatalog } from '../../models/dataset.model';
import { EntityPickerComponent, type PickedEntity } from '../../shared/components/entity-picker/entity-picker.component';
import { UnitComboboxComponent } from '../../shared/components/unit-combobox/unit-combobox.component';
import type { DatasetEditSession, ExperimentField } from './dataset-edit-session';
import { exact, pressureUnitLabel } from './measurement-worksheet.component';

const BASIS_LABELS: Record<PressureBasis, string> = {
    absolute: 'Total pressure',
    partial: 'Partial pressure',
    relative: 'Relative (p/p₀)',
};

/** What the single canonical pressure column means for each basis. */
const BASIS_HINTS: Record<PressureBasis, string> = {
    absolute: 'System total pressure, shared by every species at a point; partial pressure is y × P.',
    partial: 'Each species’ own partial pressure, as reported by the source.',
    relative: 'p/p₀ of a single species.',
};

/** Experiment conditions: read-only facts, or edit controls in edit mode. */
@Component({
    selector: 'adsmod-experiment-meta',
    standalone: true,
    imports: [DatePipe, EntityPickerComponent, UnitComboboxComponent],
    template: `
        @if (!editing()) {
            <dl class="experiment-facts">
                <div class="wide"><dt>Experiment</dt><dd [title]="experiment().external_key">{{ experiment().name }}</dd></div>
                <div class="wide"><dt>Material</dt><dd>{{ experiment().adsorbent }}</dd></div>
                <div class="wide"><dt>Species</dt><dd>{{ speciesText() }}</dd></div>
                <div><dt>Temperature</dt><dd>{{ temperatureText() }}</dd></div>
                <div><dt>Pressure basis</dt><dd [title]="basisHint(experiment().pressure_basis)">{{ basisLabel(experiment().pressure_basis) }}</dd></div>
                <div><dt>Units</dt><dd>{{ unitLabel(experiment().pressure_original_unit) }} · {{ experiment().uptake_original_unit }}</dd></div>
                @if (experiment().saturation_pressure_pa !== null) {
                    <div><dt>p₀</dt><dd>{{ exact(experiment().saturation_pressure_pa) }} Pa</dd></div>
                }
                <div><dt>Fitting</dt><dd [title]="experiment().ineligibility_reason ?? ''">{{ experiment().fitting_eligible ? 'Eligible' : 'Not eligible' }}</dd></div>
                @if (experiment().edited_at) {
                    <div><dt>Edited</dt><dd>{{ experiment().edited_at | date: 'medium' }}</dd></div>
                }
            </dl>
        } @else {
            <div class="experiment-editor" role="group" [attr.aria-label]="'Edit ' + experiment().name">
                <label class="field" [class.invalid]="issue('name')">
                    <span>Experiment name</span>
                    <input type="text" [value]="value('name', experiment().name)" (change)="set('name', $any($event.target).value, experiment().name)" />
                    @if (issue('name'); as problem) { <small role="alert">{{ problem.message }}</small> }
                </label>
                <div class="field-row">
                <div class="field" [class.invalid]="issue('adsorbent_id')">
                    <span>Material</span>
                    <div class="entity-current">
                        <strong>{{ materialName() }}</strong>
                        @if (session().isExperimentChanged(experiment().id, 'adsorbent_id')) {
                            <button class="button quiet" type="button" (click)="resetMaterial()">Undo</button>
                        }
                    </div>
                    <adsmod-entity-picker kind="material" label="Replace with" [compact]="true" (selected)="pickMaterial($event)" />
                    @if (issue('adsorbent_id'); as problem) { <small role="alert">{{ problem.message }}</small> }
                </div>
                @for (component of experiment().components; track component.position) {
                    <div class="field" [class.invalid]="issue('components')">
                        <span>Species{{ experiment().components.length > 1 ? ' ' + component.position : '' }}</span>
                        <div class="entity-current">
                            <strong>{{ speciesName(component) }}</strong>
                        </div>
                        <adsmod-entity-picker kind="species" label="Replace with" [compact]="true" (selected)="pickSpecies(component, $event)" />
                    </div>
                }
                </div>
                @if (issue('components'); as problem) { <small class="group-error" role="alert">{{ problem.message }}</small> }
                <div class="field-row">
                    <label class="field" [class.invalid]="issue('temperature_original')">
                        <span>Temperature</span>
                        <input type="text" inputmode="decimal" [value]="exact(value('temperature_original', experiment().temperature_original))" (change)="setNumber('temperature_original', $any($event.target).value, experiment().temperature_original)" />
                        @if (issue('temperature_original'); as problem) { <small role="alert">{{ problem.message }}</small> }
                    </label>
                    @if (units(); as catalog) {
                        <div class="field">
                            <span>Temperature unit</span>
                            <adsmod-unit-combobox quantity="temperature" label="Temperature unit" [showOverrideState]="false" [options]="catalog.temperature" [value]="value('temperature_original_unit', experiment().temperature_original_unit) ?? ''" (valueChange)="set('temperature_original_unit', $event, experiment().temperature_original_unit)" />
                        </div>
                    }
                </div>
                <div class="field-row">
                    <label class="field">
                        <span>Pressure basis</span>
                        <select class="select-input" [value]="value('pressure_basis', experiment().pressure_basis)" (change)="set('pressure_basis', $any($event.target).value, experiment().pressure_basis)">
                            @for (basis of bases; track basis) { <option [value]="basis">{{ basisLabel(basis) }}</option> }
                        </select>
                        <small class="hint">{{ basisHint(value('pressure_basis', experiment().pressure_basis)) }}</small>
                    </label>
                    <label class="field" [class.invalid]="issue('saturation_pressure_pa')">
                        <span>Saturation pressure p₀ [Pa]</span>
                        <input type="text" inputmode="decimal" placeholder="Not set" [value]="exact(value('saturation_pressure_pa', experiment().saturation_pressure_pa))" (change)="setNumber('saturation_pressure_pa', $any($event.target).value, experiment().saturation_pressure_pa)" />
                        @if (issue('saturation_pressure_pa'); as problem) { <small role="alert">{{ problem.message }}</small> }
                    </label>
                </div>
                @if (units(); as catalog) {
                    <div class="field-row">
                        <div class="field" [class.invalid]="issue('pressure_unit')">
                            <span>Pressure unit (all points)</span>
                            <adsmod-unit-combobox quantity="pressure" label="Pressure unit (all points)" [showOverrideState]="false" [options]="catalog.pressure" [invalid]="!!issue('pressure_unit')" [value]="value('pressure_unit', experiment().pressure_original_unit) ?? ''" (valueChange)="set('pressure_unit', $event, experiment().pressure_original_unit)" />
                            @if (issue('pressure_unit'); as problem) { <small role="alert">{{ problem.message }}</small> }
                        </div>
                        <div class="field" [class.invalid]="issue('uptake_unit')">
                            <span>Uptake unit (all points)</span>
                            <adsmod-unit-combobox quantity="uptake" label="Uptake unit (all points)" [showOverrideState]="false" [options]="catalog.uptake" [invalid]="!!issue('uptake_unit')" [value]="value('uptake_unit', experiment().uptake_original_unit) ?? ''" (valueChange)="set('uptake_unit', $event, experiment().uptake_original_unit)" />
                            @if (issue('uptake_unit'); as problem) { <small role="alert">{{ problem.message }}</small> }
                        </div>
                    </div>
                    <p class="unit-note">Changing a unit reinterprets every measurement in this experiment.</p>
                }
            </div>
        }
    `,
    styles: [
        `
            :host { display: block; min-width: 0; }
            .experiment-facts { display: grid; grid-template-columns: repeat(auto-fill, minmax(9rem, 1fr)); gap: 0.5rem 1rem; margin: 0; }
            .experiment-facts .wide { grid-column: span 2; }
            .experiment-facts dt { color: var(--text-tertiary); font-size: 0.64rem; font-weight: 700; letter-spacing: 0.03em; text-transform: uppercase; }
            .experiment-facts dd { margin: 0; overflow: hidden; font-size: 0.82rem; text-overflow: ellipsis; white-space: nowrap; }
            .experiment-editor { display: grid; gap: 0.6rem; }
            .field { display: grid; gap: 0.25rem; min-width: 0; color: var(--text-secondary); font-size: 0.72rem; font-weight: 600; }
            .field input[type='text'] { width: 100%; }
            .field.invalid input, .field.invalid select { border-color: var(--danger); }
            .field small, .group-error { color: var(--danger-text); font-size: 0.72rem; font-weight: 500; }
            .field small.hint { color: var(--text-secondary); }
            .field-row { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0.6rem; align-items: start; }
            .entity-current { display: flex; align-items: center; gap: 0.5rem; color: var(--text-primary); font-size: 0.82rem; }
            .hint { color: var(--text-tertiary); font-size: 0.72rem; font-weight: 500; }
            .unit-note { margin: 0; color: var(--text-tertiary); font-size: 0.72rem; }
            @media (max-width: 640px) {
                .field-row { grid-template-columns: 1fr; }
                .experiment-facts .wide { grid-column: span 1; }
            }
        `,
    ],
})
export class ExperimentMetaComponent {
    readonly experiment = input.required<ExperimentSummary>();
    readonly session = input.required<DatasetEditSession>();
    readonly editing = input(false);
    readonly units = input<UnitCatalog | null>(null);

    protected readonly bases: readonly PressureBasis[] = ['absolute', 'partial', 'relative'];
    protected readonly exact = exact;
    protected readonly unitLabel = pressureUnitLabel;
    /** Names of picked replacements, so the editor can show them before saving. */
    private readonly pickedNames = signal<Record<string, string>>({});

    protected readonly speciesText = computed(() =>
        this.experiment()
            .components.map((component) => component.name)
            .join(' + '),
    );
    protected readonly temperatureText = computed(() => {
        const experiment = this.experiment();
        const kelvin = `${exact(experiment.temperature_k)} K`;
        return experiment.temperature_original_unit && experiment.temperature_original_unit !== 'K'
            ? `${exact(experiment.temperature_original)} ${experiment.temperature_original_unit} (${kelvin})`
            : kelvin;
    });

    protected basisLabel(basis: PressureBasis): string {
        return BASIS_LABELS[basis];
    }

    protected basisHint(basis: PressureBasis): string {
        return BASIS_HINTS[basis];
    }

    protected value<T extends string | number | null>(field: ExperimentField, stored: T): T {
        return this.session().experimentValue<T>(this.experiment(), field, stored);
    }

    protected issue(field: string) {
        return this.session().issueFor('experiment', this.experiment().id, field);
    }

    protected set(field: ExperimentField, value: string, stored: string | number | null): void {
        this.session().setExperimentValue(this.experiment(), field, value, stored);
    }

    protected setNumber(field: ExperimentField, raw: string, stored: number | null): void {
        const text = raw.trim().replace(',', '.');
        this.session().setExperimentValue(this.experiment(), field, text === '' ? null : Number(text), stored);
    }

    protected materialName(): string {
        const id = this.value('adsorbent_id', this.experiment().adsorbent_id);
        return id === this.experiment().adsorbent_id ? this.experiment().adsorbent : (this.pickedNames()[`m${id}`] ?? `Material #${id}`);
    }

    protected pickMaterial(entity: PickedEntity | null): void {
        if (!entity) return;
        this.pickedNames.update((names) => ({ ...names, [`m${entity.id}`]: entity.name }));
        this.session().setExperimentValue(this.experiment(), 'adsorbent_id', entity.id, this.experiment().adsorbent_id);
    }

    protected resetMaterial(): void {
        this.session().setExperimentValue(this.experiment(), 'adsorbent_id', this.experiment().adsorbent_id, this.experiment().adsorbent_id);
    }

    protected speciesName(component: ExperimentComponent): string {
        const id = this.session().speciesValue(this.experiment(), component.position, component.adsorbate_id);
        return id === component.adsorbate_id ? component.name : (this.pickedNames()[`s${id}`] ?? `Species #${id}`);
    }

    protected pickSpecies(component: ExperimentComponent, entity: PickedEntity | null): void {
        if (!entity) return;
        this.pickedNames.update((names) => ({ ...names, [`s${entity.id}`]: entity.name }));
        this.session().setSpeciesValue(this.experiment(), component.position, entity.id, component.adsorbate_id);
    }
}
