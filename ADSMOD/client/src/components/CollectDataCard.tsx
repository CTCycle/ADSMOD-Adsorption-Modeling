import React, { useState } from 'react';
import { NumberInput } from './UIComponents';

export const CollectDataCard: React.FC = () => {
    const [guestFraction, setGuestFraction] = useState(1.0);
    const [hostFraction, setHostFraction] = useState(1.0);
    const [experimentsFraction, setExperimentsFraction] = useState(1.0);
    const [parallelRequests, setParallelRequests] = useState(20);

    const handleLoadFiles = () => {
        console.log('[INFO] Load data from files (Not implemented yet)');
    };

    const handleCollectData = () => {
        console.log('[INFO] Collect adsorption data (Not implemented yet)');
    };

    const handleRetrieveAdsorbates = () => {
        console.log('[INFO] Retrieve adsorbates properties (Not implemented yet)');
    };

    const handleRetrieveAdsorbents = () => {
        console.log('[INFO] Retrieve adsorbents properties (Not implemented yet)');
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
                    <NumberInput
                        label="Parallel requests"
                        value={parallelRequests}
                        onChange={setParallelRequests}
                        min={1}
                        max={500}
                        step={1}
                        precision={0}
                    />
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                    <button className="button secondary" onClick={handleLoadFiles} style={{ justifyContent: 'center' }}>
                        Load data from files
                    </button>
                    <button className="button primary" onClick={handleCollectData} style={{ justifyContent: 'center' }}>
                        Collect adsorption data
                    </button>
                    <button className="button secondary" onClick={handleRetrieveAdsorbates} style={{ justifyContent: 'center' }}>
                        Retrieve adsorbates props
                    </button>
                    <button className="button secondary" onClick={handleRetrieveAdsorbents} style={{ justifyContent: 'center' }}>
                        Retrieve adsorbents props
                    </button>
                </div>
            </div>
        </div>
    );
};
