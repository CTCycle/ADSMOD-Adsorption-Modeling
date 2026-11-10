// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, computed, input, model, output } from '@angular/core';
import type {
    ColumnDetection,
    ColumnRole,
    ImportIssue,
    ImportMapping,
    ImportPreview,
    ImportableStructure,
    PressureBasis,
    UnitCatalog,
    UnitOption,
    UnitQuantity,
} from '../../models/dataset.model';
import {
    UnitComboboxComponent,
    type CustomUnitChoice,
} from '../../shared/components/unit-combobox/unit-combobox.component';
import { unitKey } from '../../shared/utils/unit-key';

const COLUMN_ROLES: readonly { value: ColumnRole; label: string }[] = [
    { value: 'ignore', label: 'Ignore' },
    { value: 'experiment_id', label: 'Experiment ID' },
    { value: 'experiment_name', label: 'Experiment name' },
    { value: 'pressure', label: 'Pressure series' },
    { value: 'uptake', label: 'Adsorbed amount / uptake' },
    { value: 'adsorbate', label: 'Adsorbate species' },
    { value: 'adsorbate_smiles', label: 'Adsorbate SMILES' },
    { value: 'adsorbent', label: 'Adsorbent material' },
    { value: 'temperature', label: 'Temperature' },
    { value: 'pressure_unit', label: 'Pressure unit' },
    { value: 'uptake_unit', label: 'Uptake unit' },
    { value: 'temperature_unit', label: 'Temperature unit' },
    { value: 'uptake_stddev', label: 'Uptake uncertainty' },
    { value: 'gas_mole_fraction', label: 'Gas mole fraction (y)' },
    { value: 'saturation_pressure', label: 'Saturation pressure p₀' },
    { value: 'metadata', label: 'Additional metadata' },
];

const UNIT_FIELDS: readonly {
    quantity: UnitQuantity;
    label: string;
}[] = [
    { quantity: 'pressure', label: 'Pressure unit' },
    { quantity: 'uptake', label: 'Uptake unit' },
    { quantity: 'temperature', label: 'Temperature unit' },
];

const STRUCTURE_HINTS: Record<ImportableStructure, string> = {
    atomic: 'Each row is one pressure–uptake observation. Recommended.',
    aggregated:
        'Each row holds whole pressure and uptake series; the delimiter splits them.',
    mixed: 'Columns are mapped explicitly, including pressure–uptake pairs.',
};

const PRESSURE_HINTS: Record<PressureBasis, string> = {
    absolute:
        'Total pressure of the gas phase in Pa, shared by every species at a point. Map a gas mole fraction to keep partial pressures (y × P).',
    partial:
        'Each species’ own partial pressure, converted to Pa. Use only when the file reports partial pressures.',
    relative:
        'Fraction of the saturation pressure p/p₀ (0 to 1), single-species experiments only. Requires p/p₀, dimensionless or % units.',
};

const DUPLICATE_HINTS: Record<ImportMapping['duplicate_policy'], string> = {
    reject: 'A species may have one uptake per pressure; repeated pressures block the import.',
    average:
        'Repeated pressures of a species are averaged into one stored measurement; the replicate values are kept in its metadata.',
};

@Component({
    selector: 'adsmod-import-mapping-step',
    standalone: true,
    imports: [UnitComboboxComponent],
    template: `
        <div class="settings-cards">
            <section class="settings-card" aria-labelledby="card-dataset">
                <h3 id="card-dataset">Dataset information</h3>
                <label>
                    <span>Dataset name</span>
                    <input
                        class="select-input"
                        [value]="mapping().dataset_name"
                        (input)="setDatasetName($event)"
                    />
                </label>
                <label>
                    <span>Description (optional)</span>
                    <textarea
                        class="select-input"
                        rows="2"
                        maxlength="2000"
                        [value]="description()"
                        (input)="setDescription($event)"
                    ></textarea>
                </label>
                <small class="hint"
                    >Source file: {{ preview().filename }} ·
                    {{ preview().row_count }} rows ·
                    {{ preview().column_count }} columns</small
                >
            </section>

            <section class="settings-card" aria-labelledby="card-interpretation">
                <h3 id="card-interpretation">Data interpretation</h3>
                <label>
                    <span>Source structure</span>
                    <select
                        class="select-input"
                        [value]="mapping().structure"
                        (change)="setStructure($event)"
                    >
                        <option value="atomic">
                            Atomic — one observation per row
                        </option>
                        <option value="aggregated">
                            Aggregated — series inside rows
                        </option>
                        <option value="mixed">Mixed, explicitly mapped</option>
                    </select>
                    <small class="hint">{{
                        structureHints[mapping().structure]
                    }}</small>
                </label>
                @if (mapping().structure !== 'atomic') {
                    <label>
                        <span>Series delimiter</span>
                        <input
                            class="select-input"
                            maxlength="4"
                            placeholder="e.g. ; or ,"
                            [value]="mapping().series_delimiter || ''"
                            (input)="setSeriesDelimiter($event)"
                        />
                        <small class="hint"
                            >Character separating values inside a series cell.</small
                        >
                    </label>
                }
                <label class="check">
                    <input
                        type="checkbox"
                        [checked]="mapping().whole_file_grouping"
                        (change)="setWholeFileGrouping($event)"
                    />
                    <span>
                        Treat the entire file as one experiment
                        <small class="hint"
                            >Only for a single-isotherm file without an
                            identifier column.</small
                        >
                    </span>
                </label>
            </section>

            <section class="settings-card" aria-labelledby="card-units">
                <h3 id="card-units">
                    Scientific units
                    <a
                        class="learn-more"
                        href="/help#units"
                        target="_blank"
                        rel="noopener"
                        >Learn more</a
                    >
                </h3>
                @for (field of unitFields; track field.quantity) {
                    <div class="unit-field">
                        <span class="field-label">{{ field.label }}</span>
                        <adsmod-unit-combobox
                            [quantity]="field.quantity"
                            [label]="field.label"
                            [options]="optionsFor(field.quantity)"
                            [detected]="detectedUnit(field.quantity)"
                            [invalid]="!!unitIssue(field.quantity)"
                            [value]="mapping().unit_overrides[field.quantity] || ''"
                            (valueChange)="setUnit(field.quantity, $event)"
                            (customUnit)="onCustomUnit(field.quantity, $event)"
                        />
                        @if (unitIssue(field.quantity); as issue) {
                            <small class="unit-error" role="alert">
                                {{ issue.message }}
                                @if (issue.remediation) {
                                    {{ issue.remediation }}
                                }
                            </small>
                        }
                    </div>
                }
            </section>

            <section class="settings-card" aria-labelledby="card-validation">
                <h3 id="card-validation">Validation and processing</h3>
                <label>
                    <span>Pressure meaning</span>
                    <select
                        class="select-input"
                        [value]="mapping().pressure_basis"
                        (change)="setPressureBasis($event)"
                    >
                        <option value="absolute">Total pressure</option>
                        <option value="partial">Partial pressure</option>
                        <option value="relative">Relative p/p₀</option>
                    </select>
                    <small class="hint">{{
                        pressureHints[mapping().pressure_basis]
                    }}</small>
                </label>
                <label>
                    <span>Duplicate pressure points</span>
                    <select
                        class="select-input"
                        [value]="mapping().duplicate_policy"
                        (change)="setDuplicatePolicy($event)"
                    >
                        <option value="reject">Reject</option>
                        <option value="average">Average replicates</option>
                    </select>
                    <small class="hint">{{
                        duplicateHints[mapping().duplicate_policy]
                    }}</small>
                </label>
            </section>
        </div>

        @if (mapping().species_columns.length) {
        <section class="mapping-card" aria-labelledby="card-species">
            <h3 id="card-species">Species columns</h3>
            <small class="hint"
                >Use when each adsorbate has its own uptake column. Each row is
                converted to one measurement per species within a single
                experiment; an optional column gives that species' gas mole
                fraction.</small
            >
            @for (item of mapping().species_columns; track item.uptake_column) {
                <div class="wide-pair-row species-row">
                    <span>{{ item.uptake_column }}</span>
                    <input
                        class="select-input"
                        [attr.aria-label]="'Species for ' + item.uptake_column"
                        [value]="item.adsorbate"
                        (change)="renameSpeciesColumn($index, $event)"
                    />
                    <select
                        class="select-input"
                        [attr.aria-label]="'Gas mole fraction column for ' + item.adsorbate"
                        [value]="item.mole_fraction_column ?? ''"
                        (change)="setSpeciesFraction($index, $event)"
                    >
                        <option value="">No mole fraction</option>
                        @for (column of preview().columns; track column.name) {
                            <option [value]="column.name">{{ column.name }}</option>
                        }
                    </select>
                    <button
                        class="button quiet"
                        type="button"
                        (click)="removeSpeciesColumn($index)"
                    >
                        Remove
                    </button>
                </div>
            }
            <div class="wide-pair-row">
                <select
                    #speciesColumn
                    class="select-input"
                    aria-label="Uptake column of another species"
                >
                    @for (column of preview().columns; track column.name) {
                        <option [value]="column.name">{{ column.name }}</option>
                    }
                </select>
                <button
                    class="button secondary"
                    type="button"
                    (click)="addSpeciesColumn(speciesColumn.value)"
                >
                    Add species column
                </button>
            </div>
        </section>
        }

        <details class="advanced">
            <summary>
                Advanced: decimal separator, dataset-wide constants
                @if (mapping().structure !== 'atomic') {
                    and wide pairs
                }
            </summary>
            <div class="advanced-grid">
                <label>
                    <span>Decimal separator</span>
                    <select
                        class="select-input"
                        [value]="mapping().decimal_separator"
                        (change)="setDecimalSeparator($event)"
                    >
                        <option value="auto">Detect automatically</option>
                        <option value=".">Decimal point (.)</option>
                        <option value=",">Comma decimal (,)</option>
                    </select>
                    <small class="hint"
                        >Ambiguous values such as 1,234 are rejected while this
                        is automatic.</small
                    >
                </label>
                @for (constant of constantFields; track constant.key) {
                    <label>
                        <span>{{ constant.label }}</span>
                        <input
                            class="select-input"
                            [value]="mapping().constants[constant.key] || ''"
                            (input)="setConstant(constant.key, $event)"
                        />
                    </label>
                }
            </div>
            @if (mapping().structure !== 'atomic') {
                <div class="wide-pair-row">
                    <select
                        #widePressure
                        class="select-input"
                        aria-label="Wide pressure column"
                    >
                        @for (column of preview().columns; track column.name) {
                            <option [value]="column.name">
                                {{ column.name }}
                            </option>
                        }
                    </select>
                    <select
                        #wideUptake
                        class="select-input"
                        aria-label="Wide uptake column"
                    >
                        @for (column of preview().columns; track column.name) {
                            <option [value]="column.name">
                                {{ column.name }}
                            </option>
                        }
                    </select>
                    <button
                        class="button secondary"
                        type="button"
                        (click)="addWidePair(widePressure.value, wideUptake.value)"
                    >
                        Add pressure–uptake pair
                    </button>
                </div>
                @for (
                    pair of mapping().wide_pairs;
                    track pair.pressure_column + pair.uptake_column
                ) {
                    <p class="pair">
                        {{ pair.pressure_column }} ↔ {{ pair.uptake_column }}
                        <button
                            class="button quiet"
                            type="button"
                            (click)="removeWidePair($index)"
                        >
                            Remove
                        </button>
                    </p>
                }
            }
        </details>

        <section class="mapping-card" aria-labelledby="card-mapping">
            <h3 id="card-mapping">
                Column mapping
                <a
                    class="learn-more"
                    href="/help#column-mapping"
                    target="_blank"
                    rel="noopener"
                    >Learn more</a
                >
            </h3>
            <div class="mapping-wrap">
                <table class="mapping-table">
                    <thead>
                        <tr>
                            <th>Source column</th>
                            <th>Representative values</th>
                            <th>Canonical field</th>
                            <th>Experiment grouping</th>
                            <th>Detection</th>
                        </tr>
                    </thead>
                    <tbody>
                        @for (column of preview().columns; track column.name) {
                            <tr>
                                <th scope="row" data-label="Source column">
                                    {{ column.name }}
                                </th>
                                <td data-label="Representative values">
                                    {{ column.sample_values.slice(0, 3).join(', ') }}
                                </td>
                                <td data-label="Canonical field">
                                    <select
                                        class="select-input"
                                        [attr.aria-label]="'Canonical field for ' + column.name"
                                        [value]="mapping().column_roles[column.name]"
                                        (change)="setColumnRole(column.name, $event)"
                                    >
                                        @for (role of roles; track role.value) {
                                            <option
                                                [value]="role.value"
                                                [selected]="mapping().column_roles[column.name] === role.value"
                                            >
                                                {{ role.label }}
                                            </option>
                                        }
                                    </select>
                                </td>
                                <td data-label="Experiment grouping">
                                    <input
                                        type="checkbox"
                                        [checked]="mapping().grouping_columns.includes(column.name)"
                                        (change)="setGrouping(column.name, $event)"
                                        [attr.aria-label]="'Group experiments by ' + column.name"
                                    />
                                </td>
                                <td data-label="Detection">
                                    <span class="confidence-badge"
                                        >{{ percent(column.confidence) }}</span
                                    >
                                    @if (isManualRole(column)) {
                                        <span class="role-badge manual"
                                            >Manual</span
                                        >
                                    } @else {
                                        <span class="role-badge auto"
                                            >Auto</span
                                        >
                                    }
                                    @if (column.detected_unit) {
                                        <span class="detected-unit">{{
                                            column.detected_unit
                                        }}</span>
                                    }
                                    @if (column.array_like) {
                                        <span class="detected-unit"
                                            >series-like</span
                                        >
                                    }
                                </td>
                            </tr>
                        }
                    </tbody>
                </table>
            </div>
        </section>
    `,
    styles: [
        `
            :host {
                display: grid;
                gap: 1rem;
            }
            h3 {
                display: flex;
                align-items: baseline;
                justify-content: space-between;
                gap: 0.75rem;
                margin: 0 0 0.25rem;
                font-size: 0.9rem;
            }
            .learn-more {
                font-size: 0.74rem;
                font-weight: 500;
            }
            .settings-cards {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(min(100%, 26rem), 1fr));
                gap: 0.75rem;
            }
            .settings-card,
            .mapping-card,
            .advanced {
                display: grid;
                align-content: start;
                gap: 0.7rem;
                padding: 0.9rem;
                border: 1px solid var(--slate-200);
                border-radius: 12px;
                background: var(--surface);
            }
            label,
            .unit-field {
                display: grid;
                gap: 0.3rem;
                color: var(--slate-700);
                font-size: 0.78rem;
                font-weight: 650;
            }
            .field-label {
                color: var(--slate-700);
                font-size: 0.78rem;
                font-weight: 650;
            }
            .hint {
                color: var(--slate-500);
                font-size: 0.74rem;
                font-weight: 400;
                line-height: 1.35;
            }
            .unit-error {
                color: #b91c1c;
                font-size: 0.76rem;
                font-weight: 500;
            }
            label.check {
                display: flex;
                align-items: flex-start;
                gap: 0.55rem;
            }
            label.check input {
                margin-top: 0.2rem;
            }
            textarea.select-input {
                resize: vertical;
                background-image: none;
                cursor: text;
            }
            .advanced {
                gap: 0.2rem;
            }
            .advanced summary {
                cursor: pointer;
                font-size: 0.85rem;
                font-weight: 700;
            }
            .advanced-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(min(100%, 14rem), 1fr));
                gap: 0.75rem;
                margin-top: 0.8rem;
            }
            .wide-pair-row {
                display: grid;
                grid-template-columns: 1fr 1fr auto;
                gap: 0.5rem;
                margin-top: 0.8rem;
            }
            .wide-pair-row.species-row {
                grid-template-columns: 1fr 1fr 1fr auto;
            }
            .pair {
                margin: 0.4rem 0 0;
                font-size: 0.82rem;
            }
            .mapping-wrap {
                border: 1px solid var(--slate-200);
                border-radius: 10px;
                overflow-x: auto;
            }
            .mapping-table {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.78rem;
            }
            .mapping-table th,
            .mapping-table td {
                max-width: 20rem;
                padding: 0.55rem 0.7rem;
                overflow: hidden;
                border-bottom: 1px solid var(--slate-200);
                text-align: left;
                text-overflow: ellipsis;
                vertical-align: middle;
                white-space: nowrap;
            }
            .mapping-table thead th {
                position: sticky;
                top: 0;
                z-index: 1;
                background: var(--slate-50);
            }
            .mapping-table select {
                min-width: 11rem;
            }
            .confidence-badge,
            .role-badge,
            .detected-unit {
                display: inline-block;
                margin-right: 0.3rem;
                padding: 0.1rem 0.45rem;
                border-radius: 999px;
                font-size: 0.7rem;
            }
            .confidence-badge {
                color: var(--link);
                background: var(--primary-50);
            }
            .role-badge.auto {
                color: #0b6b3a;
                background: color-mix(in srgb, #16a34a 16%, transparent);
            }
            .role-badge.manual {
                color: #92400e;
                background: color-mix(in srgb, #f59e0b 22%, transparent);
            }
            .detected-unit {
                background: var(--slate-100);
            }
            @media (max-width: 720px) {
                .wide-pair-row {
                    grid-template-columns: minmax(0, 1fr);
                }
                .mapping-table thead {
                    position: absolute;
                    width: 1px;
                    height: 1px;
                    overflow: hidden;
                    clip-path: inset(50%);
                }
                .mapping-table,
                .mapping-table tbody,
                .mapping-table tr,
                .mapping-table th,
                .mapping-table td {
                    display: block;
                    max-width: none;
                }
                .mapping-table tr {
                    padding: 0.5rem 0.2rem;
                    border-bottom: 1px solid var(--slate-200);
                }
                .mapping-table th,
                .mapping-table td {
                    display: grid;
                    grid-template-columns: 8.5rem minmax(0, 1fr);
                    gap: 0.5rem;
                    align-items: center;
                    border: 0;
                    white-space: normal;
                }
                .mapping-table td::before,
                .mapping-table th::before {
                    content: attr(data-label);
                    color: var(--slate-500);
                    font-size: 0.72rem;
                    font-weight: 700;
                }
                .mapping-table select {
                    min-width: 0;
                }
            }
        `,
    ],
})
export class ImportMappingStepComponent {
    readonly preview = input.required<ImportPreview>();
    readonly unitCatalog = input<UnitCatalog | null>(null);
    readonly unitIssues = input<readonly ImportIssue[]>([]);
    readonly mapping = model.required<ImportMapping>();
    readonly description = model('');
    readonly catalogChanged = output<UnitOption>();

    protected readonly roles = COLUMN_ROLES;
    protected readonly unitFields = UNIT_FIELDS;
    protected readonly structureHints = STRUCTURE_HINTS;
    protected readonly pressureHints = PRESSURE_HINTS;
    protected readonly duplicateHints = DUPLICATE_HINTS;
    protected readonly constantFields = [
        { key: 'adsorbate', label: 'Constant adsorbate' },
        { key: 'adsorbate_smiles', label: 'Constant adsorbate SMILES' },
        { key: 'adsorbent', label: 'Constant adsorbent' },
        { key: 'temperature', label: 'Constant temperature' },
        { key: 'pressure_unit', label: 'Constant pressure unit' },
        { key: 'uptake_unit', label: 'Constant uptake unit' },
        { key: 'temperature_unit', label: 'Constant temperature unit' },
        {
            key: 'adsorbate_molar_mass_g_mol',
            label: 'Adsorbate molar mass (g/mol)',
        },
    ];

    private readonly optionsByQuantity = computed(() => {
        const catalog = this.unitCatalog();
        const sessionUnits = this.mapping().unit_definitions;
        const result = {} as Record<UnitQuantity, UnitOption[]>;
        for (const { quantity } of UNIT_FIELDS) {
            const saved = catalog?.[quantity] ?? [];
            const known = new Set(saved.map((option) => unitKey(option.symbol)));
            const session = sessionUnits
                .filter(
                    (unit) =>
                        unit.quantity === quantity &&
                        !known.has(unitKey(unit.symbol)),
                )
                .map(
                    (unit): UnitOption => ({
                        quantity,
                        symbol: unit.symbol,
                        label: unit.symbol,
                        name: 'Custom unit (this import only)',
                        group: 'This import only',
                        note:
                            unit.factor_to_canonical === null
                                ? 'Label without a conversion definition.'
                                : `User-defined factor to canonical unit: ${unit.factor_to_canonical}`,
                        aliases: [],
                        built_in: false,
                        has_conversion: unit.factor_to_canonical !== null,
                        factor_to_canonical: unit.factor_to_canonical,
                        id: null,
                    }),
                );
            result[quantity] = [...saved, ...session];
        }
        return result;
    });

    protected optionsFor(quantity: UnitQuantity): readonly UnitOption[] {
        return this.optionsByQuantity()[quantity];
    }

    protected detectedUnit(quantity: UnitQuantity): string | null {
        const roles = this.mapping().column_roles;
        const column = this.preview().columns.find(
            (item) => roles[item.name] === quantity,
        );
        return column?.detected_unit ?? null;
    }

    protected unitIssue(quantity: UnitQuantity): ImportIssue | undefined {
        return this.unitIssues().find((issue) => issue.column === quantity);
    }

    protected percent(value: number): string {
        return `${Math.round(value * 100)}%`;
    }

    protected isManualRole(column: ColumnDetection): boolean {
        return this.mapping().column_roles[column.name] !== column.proposed_role;
    }

    protected inputValue(event: Event): string {
        return (event.target as HTMLInputElement).value;
    }

    protected setDatasetName(event: Event): void {
        this.mapping.update((current) => ({
            ...current,
            dataset_name: this.inputValue(event),
        }));
    }

    protected setDescription(event: Event): void {
        this.description.set(this.inputValue(event));
    }

    protected setStructure(event: Event): void {
        this.mapping.update((current) => ({
            ...current,
            structure: (event.target as HTMLSelectElement)
                .value as ImportableStructure,
        }));
    }

    protected setPressureBasis(event: Event): void {
        this.mapping.update((current) => ({
            ...current,
            pressure_basis: (event.target as HTMLSelectElement)
                .value as PressureBasis,
        }));
    }

    protected setDuplicatePolicy(event: Event): void {
        this.mapping.update((current) => ({
            ...current,
            duplicate_policy: (event.target as HTMLSelectElement)
                .value as ImportMapping['duplicate_policy'],
        }));
    }

    protected setDecimalSeparator(event: Event): void {
        const value = (event.target as HTMLSelectElement).value;
        if (value === 'auto' || value === '.' || value === ',') {
            this.mapping.update((current) => ({
                ...current,
                decimal_separator: value,
            }));
        }
    }

    protected setSeriesDelimiter(event: Event): void {
        const value = this.inputValue(event);
        this.mapping.update((current) => ({
            ...current,
            series_delimiter: value || null,
        }));
    }

    protected setUnit(quantity: UnitQuantity, value: string): void {
        const text = value.trim();
        this.mapping.update((current) => {
            const unit_overrides = { ...current.unit_overrides };
            if (text) {
                unit_overrides[quantity] = text;
            } else {
                delete unit_overrides[quantity];
            }
            return { ...current, unit_overrides };
        });
    }

    protected onCustomUnit(quantity: UnitQuantity, choice: CustomUnitChoice): void {
        if (choice.saved) {
            this.catalogChanged.emit(choice.saved);
            return;
        }
        this.mapping.update((current) => ({
            ...current,
            unit_definitions: [
                ...current.unit_definitions.filter(
                    (unit) =>
                        unit.quantity !== quantity ||
                        unitKey(unit.symbol) !==
                            unitKey(choice.definition.symbol),
                ),
                choice.definition,
            ],
        }));
    }

    protected setConstant(role: string, event: Event): void {
        const value = this.inputValue(event).trim();
        this.mapping.update((current) => {
            const constants = { ...current.constants };
            if (value) {
                constants[role] = value;
            } else {
                delete constants[role];
            }
            return { ...current, constants };
        });
    }

    protected setColumnRole(column: string, event: Event): void {
        const role = (event.target as HTMLSelectElement).value as ColumnRole;
        this.mapping.update((current) => ({
            ...current,
            column_roles: { ...current.column_roles, [column]: role },
        }));
    }

    protected setGrouping(column: string, event: Event): void {
        const checked = (event.target as HTMLInputElement).checked;
        this.mapping.update((current) => ({
            ...current,
            grouping_columns: checked
                ? [...new Set([...current.grouping_columns, column])]
                : current.grouping_columns.filter((item) => item !== column),
        }));
    }

    protected setWholeFileGrouping(event: Event): void {
        const checked = (event.target as HTMLInputElement).checked;
        this.mapping.update((current) => ({
            ...current,
            whole_file_grouping: checked,
            grouping_columns: checked ? [] : current.grouping_columns,
        }));
    }

    protected addWidePair(pressure: string, uptake: string): void {
        if (!pressure || !uptake || pressure === uptake) {
            return;
        }
        this.mapping.update((current) => ({
            ...current,
            wide_pairs: [
                ...current.wide_pairs,
                { pressure_column: pressure, uptake_column: uptake },
            ],
        }));
    }

    protected addSpeciesColumn(column: string): void {
        if (!column) {
            return;
        }
        this.mapping.update((current) =>
            current.species_columns.some((item) => item.uptake_column === column)
                ? current
                : {
                      ...current,
                      column_roles: { ...current.column_roles, [column]: 'ignore' },
                      species_columns: [
                          ...current.species_columns,
                          { uptake_column: column, adsorbate: column },
                      ],
                  },
        );
    }

    protected renameSpeciesColumn(index: number, event: Event): void {
        const adsorbate = (event.target as HTMLInputElement).value.trim();
        if (!adsorbate) {
            return;
        }
        this.mapping.update((current) => ({
            ...current,
            species_columns: current.species_columns.map((item, itemIndex) =>
                itemIndex === index ? { ...item, adsorbate } : item,
            ),
        }));
    }

    protected setSpeciesFraction(index: number, event: Event): void {
        const column = (event.target as HTMLSelectElement).value || null;
        this.mapping.update((current) => ({
            ...current,
            species_columns: current.species_columns.map((item, itemIndex) =>
                itemIndex === index ? { ...item, mole_fraction_column: column } : item,
            ),
        }));
    }

    protected removeSpeciesColumn(index: number): void {
        this.mapping.update((current) => ({
            ...current,
            species_columns: current.species_columns.filter(
                (_, itemIndex) => itemIndex !== index,
            ),
        }));
    }

    protected removeWidePair(index: number): void {
        this.mapping.update((current) => ({
            ...current,
            wide_pairs: current.wide_pairs.filter(
                (_, itemIndex) => itemIndex !== index,
            ),
        }));
    }
}
