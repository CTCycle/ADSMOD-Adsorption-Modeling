import React from 'react';

export const MachineLearningPage: React.FC = () => {
    return (
        <div className="ml-page" style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
            <div style={{ maxWidth: '600px', margin: '0 auto', border: '1px dashed var(--border-color)', borderRadius: '12px', padding: '4rem 2rem' }}>
                <div style={{ fontSize: '4rem', marginBottom: '1rem' }}>🧠</div>
                <h2 style={{ color: 'var(--text-primary)', marginBottom: '0.5rem' }}>Machine Learning Analysis</h2>
                <p>Advanced analysis and prediction tools are coming soon.</p>
                <div style={{ marginTop: '2rem' }}>
                    <span className="sc-badge">Under Construction</span>
                </div>
            </div>
        </div>
    );
};
