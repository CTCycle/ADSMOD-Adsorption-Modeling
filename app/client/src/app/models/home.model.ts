// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

export interface HomeSourceCount {
    key: string;
    name: string;
    record_count: number;
}

export interface HomeSummary {
    datasets: number;
    experiments: number;
    species: number;
    materials: number;
    sources: HomeSourceCount[];
    fitting_sessions: number;
    fitting_sessions_by_status: Record<string, number>;
    latest_fitting_at: string | null;
}
