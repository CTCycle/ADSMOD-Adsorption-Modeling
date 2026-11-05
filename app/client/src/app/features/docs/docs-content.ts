// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

export interface DocsStep {
    title: string;
    description: string;
}

export interface DocsSection {
    id: string;
    title: string;
    intro: string;
    paragraphs?: readonly string[];
    steps?: readonly DocsStep[];
    tips?: readonly string[];
}

export interface DocsGroup {
    id: string;
    label: string;
    sections: readonly DocsSection[];
}

/** Documentation anchor opened for each application page. */
export const DOCS_ANCHOR_BY_PAGE = {
    datasets: 'dataset-management',
    sources: 'sources',
    dashboards: 'dashboards',
    fitting: 'fitting',
    training: 'training',
    settings: 'settings',
} as const;

export const DOCS_GROUPS: readonly DocsGroup[] = [
    {
        id: 'start',
        label: 'Getting started',
        sections: [
            {
                id: 'getting-started',
                title: 'Getting started',
                intro: 'ADSMOD turns measured adsorption data into validated, unit-consistent isotherms that can be explored, fitted, and used for training.',
                steps: [
                    { title: 'Add data', description: 'Open Datasets and choose Add dataset to import a CSV or Excel file with the import wizard.' },
                    { title: 'Inspect it', description: 'Open a dataset to review its experiments, observations, saved column mapping, and a plot of each isotherm.' },
                    { title: 'Analyse it', description: 'Use Dashboards to compare stored isotherms, Fitting to compare model equations, and Training to prepare machine learning workflows.' },
                ],
                tips: ['External providers are discovered and acquired from the Sources page.'],
            },
        ],
    },
    {
        id: 'importing',
        label: 'Importing data',
        sections: [
            {
                id: 'importing-datasets',
                title: 'Importing datasets',
                intro: 'The import wizard maps scientific meaning and units before anything is saved. Nothing is written to the database until the last step.',
                steps: [
                    { title: 'Preview', description: 'Check the detected structure, row and column counts, and the first rows of the file.' },
                    { title: 'Map columns', description: 'Review dataset information, data interpretation, scientific units, validation settings, and the column mapping. Detected values are pre-filled and marked Auto; anything you change is marked Manual.' },
                    { title: 'Validate series', description: 'Review each experiment, its observations, and any warnings or errors. Datasets with blocking errors cannot be saved.' },
                    { title: 'Save', description: 'Save the validated dataset. Source values and units are retained alongside the canonical values.' },
                ],
                tips: [
                    'Atomic layout (one observation per row) is the most reliable format. Aggregated layouts are accepted when pressure and uptake series can be paired unambiguously.',
                    'Your selections are kept when you move Back and forward between steps, but they are lost if you close the wizard.',
                ],
            },
            {
                id: 'column-mapping',
                title: 'Column mapping',
                intro: 'Each source column is assigned a canonical field such as pressure, uptake, temperature, adsorbate, or adsorbent.',
                paragraphs: [
                    'The detection column shows the confidence of the automatic proposal and any unit found in the header, for example Pressure [bar]. Choosing a different canonical field marks the column as a manual mapping.',
                    'Select one or more grouping columns to define what makes up one experiment (for example an experiment ID). Use "Treat the entire file as one experiment" only for a single-isotherm file without an identifier.',
                    'Values that are the same for the whole file, such as the adsorbate or temperature, can be supplied as dataset-wide constants under Advanced.',
                ],
                tips: ['Map a per-row unit column (pressure_unit, uptake_unit, temperature_unit) when units differ between rows.'],
            },
        ],
    },
    {
        id: 'science',
        label: 'Units and validation',
        sections: [
            {
                id: 'units',
                title: 'Scientific units and conversions',
                intro: 'Every value is converted to a canonical unit for calculations (pressure to Pa, uptake to mol/kg, temperature to K) while the original value and unit are kept.',
                paragraphs: [
                    'Unit fields are searchable. Search by symbol, name, or a common alias, or type a unit directly. A unit already in the catalog is selected automatically.',
                    'Mass-based uptake units (mg/g, g/g, wt%) need the adsorbate molar mass, which is known for common gases or can be set as a constant. Volumetric uptake units (cm³(STP)/g, cm³(STP)/kg) assume standard conditions of 0 °C and 1 atm (22.414 L/mol).',
                    'Relative pressure (p/p₀) requires a dimensionless or percent unit. Pressure meaning (absolute, partial, relative) decides how pressure is interpreted.',
                    'When you type a unit that is not in the catalog you can save it for future imports or use it for the current import only. A custom unit has no conversion until you give it a factor to the canonical unit; without one it is only a label. ADSMOD never guesses a conversion: an import that needs an undefined conversion is blocked with an explanation.',
                    'Saved custom units are stored in the database and remain available after a restart. Case and spacing are ignored when checking for duplicates, but distinct units such as mmol/g and mmol/kg are never merged.',
                ],
                tips: ['A manual unit override replaces the unit detected in the column header and is reported as a warning when they differ.'],
            },
            {
                id: 'validation',
                title: 'Experiment validation',
                intro: 'Each experiment is validated before saving. Experiments are shown as collapsible sections with their temperature, point count, pressure range, and status.',
                paragraphs: [
                    'Errors block saving. Typical causes are a missing adsorbent or temperature, unequal pressure and uptake series, an unsupported unit, or fewer than two valid observations.',
                    'Warnings and review items do not block saving on their own. Repeated pressures must be reviewed: choose to reject them, average them when fitting, or keep them as replicates, then confirm.',
                    'The wizard preview shows the first observations of each experiment. The complete series is available on the dataset page after saving.',
                ],
            },
        ],
    },
    {
        id: 'manage',
        label: 'Datasets and data sources',
        sections: [
            {
                id: 'dataset-management',
                title: 'Datasets',
                intro: 'Manage local adsorption datasets together with the adsorbent materials and chemical species they reference.',
                steps: [
                    { title: 'Add a dataset', description: 'Choose Add dataset to upload a CSV or Excel file. The import wizard previews the file, maps columns and units, groups measurements into experiments, and validates everything before saving.' },
                    { title: 'Explore a dataset', description: 'A dataset groups one or more experiments; each experiment has one material, one or more species, one temperature, and its measurements. Explore opens the worksheet view of every experiment.' },
                    { title: 'Edit values', description: 'Choose Edit dataset to change metadata, experiment conditions, materials, species, units, or individual measurements. Changes stay local until Save changes; Cancel discards them.' },
                    { title: 'Inspect materials and species', description: 'Use the Materials and Species tabs to search registered entities, see which datasets use them, and delete entities that are no longer referenced.' },
                ],
                tips: [
                    'Saving overwrites the stored values and updates the dataset\'s last update time; there is no version history.',
                    'Canonical pressure (Pa or p/p₀) and uptake (mol/kg) are recomputed from the edited source values on save.',
                    'Deleting a dataset removes its experiments, measurements, and fitting runs, but keeps materials and species.',
                ],
            },
            {
                id: 'sources',
                title: 'Sources',
                intro: 'Discover the external providers ADSMOD integrates and acquire their records into the local database.',
                steps: [
                    { title: 'Browse the catalog', description: 'Providers are grouped by the data they offer: adsorption isotherms, chemical identity, and crystal structures.' },
                    { title: 'Acquire records', description: 'Use NIST acquisition, PubChem resolution, or COD structure import. Acquired isotherms appear as the NIST ISODB dataset on the Datasets page.' },
                    { title: 'Track runs', description: 'Acquisition runs execute in the background; review progress, failures, and history from the Acquire view.' },
                ],
                tips: [
                    'Re-downloading NIST data asks before overwriting experiments you edited locally.',
                    'External provider outages do not remove locally stored records.',
                ],
            },
            {
                id: 'dashboards',
                title: 'Dashboards',
                intro: 'Explore every stored isotherm across datasets. Charts are computed from the database on demand; no aggregated copy is created.',
                steps: [
                    { title: 'Review the overview', description: 'See how measurements are distributed across temperatures, sources, materials, adsorbates, and pressure bases.' },
                    { title: 'Build a query', description: 'In Explore & compare, filter by source, pressure basis, temperature, material, adsorbate, or dataset, and optionally select individual isotherms.' },
                    { title: 'Choose an analysis', description: 'Plot isotherms coloured by temperature or category, compare uptake at one pressure, or inspect a distribution.' },
                    { title: 'Inspect a series', description: 'Select a curve or row to see its provenance, material, and adsorbate details.' },
                ],
                tips: [
                    'Uptake at a fixed pressure uses a measurement when one matches, otherwise interpolates in log pressure between bracketing points of the adsorption branch; it never extrapolates.',
                    'The page address stores the query, so a view can be bookmarked and is always recomputed from current data.',
                ],
            },
        ],
    },
    {
        id: 'modelling',
        label: 'Fitting and training',
        sections: [
            {
                id: 'fitting',
                title: 'Adsorption fitting',
                intro: 'Configure an adsorption fit and compare enabled model equations against a selected dataset.',
                steps: [
                    { title: 'Choose a dataset', description: 'Select a workspace or NIST dataset from the Dataset control.' },
                    { title: 'Set the run options', description: 'Choose the maximum iterations and optimization method for the fit.' },
                    { title: 'Choose models', description: 'Enable or disable the adsorption models you want to compare.' },
                    { title: 'Start fitting', description: 'Use Start fitting to run the workflow and follow its progress in the fitting log.' },
                ],
                tips: ['Reset log clears the visible fitting messages.', 'At least one model should remain enabled for a meaningful comparison.'],
            },
            {
                id: 'training',
                title: 'Training',
                intro: 'Prepare data, configure training, review checkpoints, and monitor training runs.',
                steps: [
                    { title: 'Prepare data', description: 'Use Data processing to select and prepare the datasets needed for training.' },
                    { title: 'Configure a run', description: 'Use the training setup view to choose the dataset, model, and run parameters.' },
                    { title: 'Monitor progress', description: 'Use the dashboard to follow active metrics and review completed runs.' },
                    { title: 'Review checkpoints', description: 'Open Checkpoints to inspect saved training artifacts before reusing them.' },
                ],
                tips: ['Training requires the optional machine learning dependencies.', 'Keep the status and metric panels visible while a run is active.'],
            },
            {
                id: 'settings',
                title: 'Settings',
                intro: 'Manage application defaults that affect future requests and jobs.',
                steps: [
                    { title: 'Choose a category', description: 'Use the tabs to move between runtime, import, public-data, fitting, and training settings.' },
                    { title: 'Save or discard edits', description: 'Save applies validated changes; Cancel restores the last saved values.' },
                    { title: 'Restart when prompted', description: 'Host and port changes are stored immediately and take effect after ADSMOD restarts.' },
                ],
                tips: ['Database connection values and storage paths remain bootstrap settings in settings/.env.', 'A running job keeps the settings snapshot it started with.'],
            },
        ],
    },
    {
        id: 'help',
        label: 'Help',
        sections: [
            {
                id: 'troubleshooting',
                title: 'Troubleshooting',
                intro: 'Common import problems and how to resolve them.',
                steps: [
                    { title: 'The unit is blocked', description: 'A selected unit has no conversion definition. Choose a catalog unit, define a conversion factor for the custom unit, or convert the source values.' },
                    { title: 'Ambiguous numbers', description: 'Values such as 1,234 can be a thousand or 1.234. Set the decimal separator under Advanced instead of leaving it on automatic.' },
                    { title: 'Unequal series lengths', description: 'In aggregated files each row needs the same number of pressure and uptake values. Check the series delimiter.' },
                    { title: 'Repeated pressures', description: 'Review the duplicate pressure policy and confirm the review item on the Validate series step.' },
                ],
            },
            {
                id: 'tips',
                title: 'Tips and best practices',
                intro: 'Habits that keep datasets scientifically reliable.',
                tips: [
                    'Prefer one observation per row with explicit unit columns or headers such as Pressure [bar].',
                    'Keep the original source file; the saved dataset records its checksum and mapping for traceability.',
                    'Give datasets descriptive names and tags so they are easy to find in Fitting and Dashboards.',
                    'Check the experiment pressure range and point count on the dataset page before fitting.',
                ],
            },
        ],
    },
];
