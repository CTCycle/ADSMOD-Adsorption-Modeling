// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import type { DatasetSummary } from '../../models/dataset.model';
import type { IconName } from '../components/icon/icon.component';

export type DatasetSource = DatasetSummary['source'];

/** Single source of truth for how a dataset origin is named and drawn. */
export const DATASET_SOURCES: Record<DatasetSource, { label: string; icon: IconName }> = {
    uploaded: { label: 'Uploaded file', icon: 'file' },
    nist: { label: 'NIST ISODB', icon: 'database' },
};
