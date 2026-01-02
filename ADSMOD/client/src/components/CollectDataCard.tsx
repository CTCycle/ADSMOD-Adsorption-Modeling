import React, { useCallback, useEffect, useState } from 'react';
import { NumberInput } from './UIComponents';
import { fetchNistData, fetchNistProperties, fetchNistStatus } from '../services';
import type { NISTStatusResponse } from '../types';

interface NistCardProps {
    onStatusUpdate: (message: string) => void;
}

/** Shared NIST status hook */
function useNistStatus() {
    const [nistStatus, setNistStatus] = useState<NISTStatusResponse | null>(null);
    const [nistStatusError, setNistStatusError] = useState<string | null>(null);
    const [isStatusLoading, setIsStatusLoading] = useState(false);

    const loadNistStatus = useCallback(async () => {
        setIsStatusLoading(true);
        const result = await fetchNistStatus();
        if (result.error) {
            setNistStatus(null);
            setNistStatusError(result.error);
        } else if (result.data) {
            setNistStatus(result.data);
            setNistStatusError(null);
        } else {
            setNistStatus(null);
            setNistStatusError('NIST status returned an empty response.');
        }
        setIsStatusLoading(false);
    }, []);

    useEffect(() => {
        void loadNistStatus();
    }, [loadNistStatus]);

    return { nistStatus, nistStatusError, isStatusLoading, loadNistStatus };
}

/**
 * NistCollectCard: Contains the "Collect adsorption data" form.
 * Handles fraction inputs and the collect button.
 */
export const NistCollectCard: React.FC<NistCardProps> = ({ onStatusUpdate }) => {
    const [guestFraction, setGuestFraction] = useState(1.0);
    const [hostFraction, setHostFraction] = useState(1.0);
    const [experimentsFraction, setExperimentsFraction] = useState(1.0);
    const [isCollecting, setIsCollecting] = useState(false);

    const { loadNistStatus } = useNistStatus();

    const handleCollectData = async () => {
        if (isCollecting) return;
        setIsCollecting(true);
        onStatusUpdate('[INFO] Starting NIST-A data collection...');
        try {
            const result = await fetchNistData({
                experiments_fraction: experimentsFraction,
                guest_fraction: guestFraction,
                host_fraction: hostFraction,
            });
            if (result.error) {
                onStatusUpdate(`[ERROR] ${result.error}`);
            } else if (result.data) {
                const lines = [
                    '[INFO] NIST-A data collection complete.',
                    '',
                    `- Experiments fetched: ${result.data.experiments_count}`,
                    `- Single-component rows: ${result.data.single_component_rows}`,
                    `- Binary-mixture rows: ${result.data.binary_mixture_rows}`,
                    `- Adsorbates rows: ${result.data.guest_rows}`,
                    `- Adsorbents rows: ${result.data.host_rows}`,
                    '',
                    'Use the Database Browser page to inspect stored tables.',
                ];
                onStatusUpdate(lines.join('\n'));
            } else {
                onStatusUpdate('[ERROR] NIST collection returned an empty response.');
            }
        } finally {
            setIsCollecting(false);
            await loadNistStatus();
        }
    };

    return (
        <div className="card">
            <div className="card-content">
                <div className="section-heading">
                    <div className="section-title">Collect adsorption data</div>
                </div>
                <div className="nist-inputs">
                    <NumberInput
                        label="Guest fraction"
                        value={guestFraction}
                        onChange={setGuestFraction}
                        min={0.001}
                        max={1.0}
                        step={0.01}
                        precision={3}
                    />
                    <NumberInput
                        label="Host fraction"
                        value={hostFraction}
                        onChange={setHostFraction}
                        min={0.001}
                        max={1.0}
                        step={0.01}
                        precision={3}
                    />
                    <NumberInput
                        label="Experiments fraction"
                        value={experimentsFraction}
                        onChange={setExperimentsFraction}
                        min={0.001}
                        max={1.0}
                        step={0.01}
                        precision={3}
                    />
                </div>

                <div className="nist-actions">
                    <button
                        className="button primary"
                        onClick={handleCollectData}
                        style={{ justifyContent: 'center' }}
                        disabled={isCollecting}
                    >
                        {isCollecting ? 'Collecting...' : 'Collect adsorption data'}
                    </button>
                </div>
            </div>
        </div>
    );
};

/**
 * NistPropertiesCard: Contains the "Enrich materials properties" section.
 * Handles property retrieval for adsorbates and adsorbents.
 */
export const NistPropertiesCard: React.FC<NistCardProps> = ({ onStatusUpdate }) => {
    const [isGuestUpdating, setIsGuestUpdating] = useState(false);
    const [isHostUpdating, setIsHostUpdating] = useState(false);
    const { nistStatus, nistStatusError, isStatusLoading } = useNistStatus();

    const isBusy = isGuestUpdating || isHostUpdating;
    const guestRows = nistStatus?.guest_rows ?? 0;
    const hostRows = nistStatus?.host_rows ?? 0;
    const dataAvailable = Boolean(nistStatus?.data_available);
    const guestAvailable = guestRows > 0;
    const hostAvailable = hostRows > 0;
    const canFetchGuest = !isBusy && guestAvailable && !isStatusLoading && !nistStatusError;
    const canFetchHost = !isBusy && hostAvailable && !isStatusLoading && !nistStatusError;

    let statusMessage = 'Collect adsorption data to enable property enrichment.';
    if (isStatusLoading) {
        statusMessage = 'Checking the local database for NIST-A materials...';
    } else if (nistStatusError) {
        statusMessage = 'Run a NIST-A collection to initialize the local database.';
    } else if (dataAvailable) {
        statusMessage = `Adsorbates: ${guestRows} | Adsorbents: ${hostRows}`;
    }

    const handleRetrieveAdsorbates = async () => {
        if (isBusy) return;
        setIsGuestUpdating(true);
        onStatusUpdate('[INFO] Retrieving adsorbates properties from PubChem...');
        const result = await fetchNistProperties({ target: 'guest' });
        if (result.error) {
            onStatusUpdate(`[ERROR] ${result.error}`);
        } else if (result.data) {
            const lines = [
                '[INFO] Adsorbates properties updated.',
                '',
                `- Names requested: ${result.data.names_requested}`,
                `- Names matched: ${result.data.names_matched}`,
                `- Rows updated: ${result.data.rows_updated}`,
            ];
            onStatusUpdate(lines.join('\n'));
        } else {
            onStatusUpdate('[ERROR] Adsorbates properties returned an empty response.');
        }
        setIsGuestUpdating(false);
    };

    const handleRetrieveAdsorbents = async () => {
        if (isBusy) return;
        setIsHostUpdating(true);
        onStatusUpdate('[INFO] Retrieving adsorbents properties from PubChem...');
        const result = await fetchNistProperties({ target: 'host' });
        if (result.error) {
            onStatusUpdate(`[ERROR] ${result.error}`);
        } else if (result.data) {
            const lines = [
                '[INFO] Adsorbents properties updated.',
                '',
                `- Names requested: ${result.data.names_requested}`,
                `- Names matched: ${result.data.names_matched}`,
                `- Rows updated: ${result.data.rows_updated}`,
            ];
            onStatusUpdate(lines.join('\n'));
        } else {
            onStatusUpdate('[ERROR] Adsorbents properties returned an empty response.');
        }
        setIsHostUpdating(false);
    };

    return (
        <div className="card nist-properties-card">
            <div className="card-content">
                <div className="nist-properties-header">
                    <div className="section-heading">
                        <div className="section-title">Enrich materials properties</div>
                        <div className="section-caption">
                            Use stored NIST-A materials to fetch PubChem properties.
                        </div>
                    </div>
                </div>
                <div className="nist-status-message">{statusMessage}</div>

                <div className="nist-actions">
                    <button
                        className="button secondary"
                        onClick={handleRetrieveAdsorbates}
                        style={{ justifyContent: 'center' }}
                        disabled={!canFetchGuest}
                    >
                        {isGuestUpdating ? 'Retrieving adsorbates...' : 'Retrieve adsorbates props'}
                    </button>
                    <button
                        className="button secondary"
                        onClick={handleRetrieveAdsorbents}
                        style={{ justifyContent: 'center' }}
                        disabled={!canFetchHost}
                    >
                        {isHostUpdating ? 'Retrieving adsorbents...' : 'Retrieve adsorbents props'}
                    </button>
                </div>
            </div>
        </div>
    );
};

/**
 * Legacy CollectDataCard - kept for backward compatibility.
 * Renders both cards stacked vertically as before.
 */
export const CollectDataCard: React.FC<NistCardProps> = ({ onStatusUpdate }) => {
    return (
        <div className="nist-card-stack">
            <NistCollectCard onStatusUpdate={onStatusUpdate} />
            <NistPropertiesCard onStatusUpdate={onStatusUpdate} />
        </div>
    );
};
