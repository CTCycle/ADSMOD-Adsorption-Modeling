import React from 'react';

interface TrainingSetupRowProps {
    onNewTrainingClick: () => void;
    onResumeTrainingClick: () => void;
    datasetAvailable: boolean;
    checkpointsAvailable: boolean;
    isTraining: boolean;
}

export const TrainingSetupRow: React.FC<TrainingSetupRowProps> = ({
    onNewTrainingClick,
    onResumeTrainingClick,
    datasetAvailable,
    checkpointsAvailable,
    isTraining,
}) => {
    const newTrainingDisabled = !datasetAvailable || isTraining;
    const resumeDisabled = true;

    const newTrainingStatus = isTraining
        ? 'Training active'
        : datasetAvailable
            ? 'Dataset ready'
            : 'Dataset required';
    const newTrainingStatusClass = isTraining ? 'busy' : datasetAvailable ? 'ready' : 'blocked';

    const resumeStatus = isTraining
        ? 'Training active'
        : checkpointsAvailable
            ? 'Checkpoints detected'
            : 'No checkpoints yet';
    const resumeStatusClass = isTraining ? 'busy' : checkpointsAvailable ? 'ready' : 'blocked';

    return (
        <div className="training-setup-row">
            <div className={`training-setup-card ${newTrainingDisabled ? 'disabled' : ''}`}>
                <div className="training-setup-card-title" style={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span className="training-setup-card-icon">🚀</span>
                        <span>Start New Training</span>
                    </div>
                    <div className={`training-setup-card-status ${newTrainingStatusClass}`} style={{ margin: 0 }}>
                        {newTrainingStatus}
                    </div>
                </div>
                <p className="training-setup-card-description">
                    Configure model, dataset, and training parameters for a fresh training run.
                </p>
                <div className="training-setup-card-action">
                    <button className="primary" onClick={onNewTrainingClick} disabled={newTrainingDisabled}>
                        New Training
                    </button>
                </div>
            </div>

            <div className={`training-setup-card ${resumeDisabled ? 'disabled' : ''}`}>
                <div className="training-setup-card-title" style={{ display: 'flex', justifyContent: 'space-between', width: '100%', alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                        <span className="training-setup-card-icon">📂</span>
                        <span>Resume Training</span>
                    </div>
                    <div className={`training-setup-card-status ${resumeStatusClass}`} style={{ margin: 0 }}>
                        {resumeStatus}
                    </div>
                </div>
                <p className="training-setup-card-description">
                    Continue training from a previously saved checkpoint with additional epochs.
                </p>
                <div className="training-setup-card-action">
                    <button className="secondary" onClick={onResumeTrainingClick} disabled={resumeDisabled}>
                        Resume Training
                    </button>
                </div>
            </div>
        </div>
    );
};

export default TrainingSetupRow;
