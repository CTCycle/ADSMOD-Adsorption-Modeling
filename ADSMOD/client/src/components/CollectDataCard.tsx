import React, { useCallback, useEffect, useState } from 'react';
import { NumberInput } from './UIComponents';
import { fetchNistData, fetchNistProperties, fetchNistStatus } from '../services';
import type { NISTStatusResponse } from '../types';

interface CollectDataCardProps {
    onStatusUpdate: (message: string) => void;
}

export const CollectDataCard: React.FC<CollectDataCardProps> = ({ onStatusUpdate }) => {
    const [guestFraction, setGuestFraction] = useState(1.0);
    const [hostFraction, setHostFraction] = useState(1.0);
    const [experimentsFraction, setExperimentsFraction] = useState(1.0);
    const [isCollecting, setIsCollecting] = useState(false);
    const [isGuestUpdating, setIsGuestUpdating] = useState(false);
    const [isHostUpdating, setIsHostUpdating] = useState(false);
    const [nistStatus, setNistStatus] = useState<NISTStatusResponse | null>(null);
    const [nistStatusError, setNistStatusError] = useState<string | null>(null);
    const [isStatusLoading, setIsStatusLoading] = useState(false);

    const isBusy = isCollecting || isGuestUpdating || isHostUpdating;

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

    const guestRows = nistStatus?.guest_rows ?? 0;
    const hostRows = nistStatus?.host_rows ?? 0;
    const dataAvailable = Boolean(nistStatus?.data_available);
    const guestAvailable = guestRows > 0;
    const hostAvailable = hostRows > 0;
    const canFetchGuest = !isBusy && guestAvailable && !isStatusLoading && !nistStatusError;
    const canFetchHost = !isBusy && hostAvailable && !isStatusLoading && !nistStatusError;

    let statusLabel = 'Not ready';
    if (isStatusLoading) {
        statusLabel = 'Checking';
    } else if (nistStatusError) {
        statusLabel = 'Unavailable';
    } else if (dataAvailable) {
        statusLabel = 'Ready';
    }

    let statusMessage = 'Collect adsorption data to enable property enrichment.';
    if (isStatusLoading) {
        statusMessage = 'Checking the local database for NIST-A materials...';
    } else if (nistStatusError) {
        statusMessage = 'Run a NIST-A collection to initialize the local database.';
    } else if (dataAvailable) {
        statusMessage = `Adsorbates: ${guestRows} | Adsorbents: ${hostRows}`;
    }

    const handleCollectData = async () => {
        if (isBusy) return;
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
        <div className="nist-card-stack">
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
                            disabled={isBusy}
                        >
                            {isCollecting ? 'Collecting...' : 'Collect adsorption data'}
                        </button>
                    </div>
                </div>
            </div>

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
                    <div className="nist-status-footer">
                        <div className="nist-status-indicator">
                            <span
                                className={`nist-status-led ${dataAvailable ? 'available' : 'unavailable'}`}
                            />
                            <span className="nist-status-label">{statusLabel}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};
