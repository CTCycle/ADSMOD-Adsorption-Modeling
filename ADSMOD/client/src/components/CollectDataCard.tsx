import React, { useState } from 'react';
import { NumberInput } from './UIComponents';
import { fetchNistData, fetchNistProperties } from '../services';

interface CollectDataCardProps {
    onStatusUpdate: (message: string) => void;
}

const DEFAULT_DATASET_NAME = 'NIST-A';

export const CollectDataCard: React.FC<CollectDataCardProps> = ({ onStatusUpdate }) => {
    const [guestFraction, setGuestFraction] = useState(1.0);
    const [hostFraction, setHostFraction] = useState(1.0);
    const [experimentsFraction, setExperimentsFraction] = useState(1.0);
    const [isCollecting, setIsCollecting] = useState(false);
    const [isGuestUpdating, setIsGuestUpdating] = useState(false);
    const [isHostUpdating, setIsHostUpdating] = useState(false);

    const isBusy = isCollecting || isGuestUpdating || isHostUpdating;

    const handleCollectData = async () => {
        if (isBusy) return;
        setIsCollecting(true);
        onStatusUpdate('[INFO] Starting NIST-A data collection...');
        const result = await fetchNistData({
            dataset_name: DEFAULT_DATASET_NAME,
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
                `- Dataset name: ${result.data.dataset_name}`,
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
        setIsCollecting(false);
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
        <div className="card">
            <div className="card-content">
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginBottom: '1.5rem' }}>
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

                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>

                    <button
                        className="button primary"
                        onClick={handleCollectData}
                        style={{ justifyContent: 'center' }}
                        disabled={isBusy}
                    >
                        {isCollecting ? 'Collecting...' : 'Collect adsorption data'}
                    </button>
                    <button
                        className="button secondary"
                        onClick={handleRetrieveAdsorbates}
                        style={{ justifyContent: 'center' }}
                        disabled={isBusy}
                    >
                        {isGuestUpdating ? 'Retrieving adsorbates...' : 'Retrieve adsorbates props'}
                    </button>
                    <button
                        className="button secondary"
                        onClick={handleRetrieveAdsorbents}
                        style={{ justifyContent: 'center' }}
                        disabled={isBusy}
                    >
                        {isHostUpdating ? 'Retrieving adsorbents...' : 'Retrieve adsorbents props'}
                    </button>
                </div>
            </div>
        </div>
    );
};
