import React, { useCallback } from 'react';
import { Switch } from './UIComponents';
import { EquationRenderer } from './EquationRenderer';
import { ModelConfigForm } from './ModelConfigForm';
import type { AdsorptionModel } from '../adsorptionModels';
import type { ModelParameters } from '../types';

interface ModelCardProps {
    model: AdsorptionModel;
    isExpanded: boolean;
    isEnabled: boolean;
    currentConfig: ModelParameters;
    onToggle: (id: string) => void;
    onEnabledChange: (modelName: string, enabled: boolean) => void;
    onConfigChange: (modelName: string, config: ModelParameters) => void;
}

/**
 * ModelCard: Displays an adsorption model in a collapsible card format.
 * 
 * Collapsed state: Shows model name, description, and rendered equation.
 * Expanded state: Shows only the configuration form for parameter bounds.
 * 
 * This design ensures description and equation are hidden when expanded,
 * and restored when collapsed again.
 */
export const ModelCard: React.FC<ModelCardProps> = ({
    model,
    isExpanded,
    isEnabled,
    currentConfig,
    onToggle,
    onEnabledChange,
    onConfigChange,
}) => {
    const handleEnabledToggle = useCallback(
        (checked: boolean) => {
            onEnabledChange(model.name, checked);
        },
        [model.name, onEnabledChange]
    );

    const handleConfigChange = useCallback(
        (config: ModelParameters) => {
            onConfigChange(model.name, config);
        },
        [model.name, onConfigChange]
    );

    const handleSwitchClick = useCallback((e: React.MouseEvent) => {
        e.stopPropagation();
    }, []);

    const handleHeaderClick = useCallback(() => {
        if (isEnabled) {
            onToggle(model.id);
        }
    }, [isEnabled, model.id, onToggle]);

    const handleKeyDown = useCallback(
        (e: React.KeyboardEvent) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                handleHeaderClick();
            }
        },
        [handleHeaderClick]
    );

    const cardId = `model-card-${model.id}`;
    const contentId = `model-content-${model.id}`;

    return (
        <div
            className={`model-grid-card ${isExpanded ? 'expanded' : ''} ${!isEnabled ? 'disabled' : ''}`}
            id={cardId}
        >
            {/* Card Header - Always visible */}
            <div
                className="model-card-header"
                onClick={handleHeaderClick}
                role="button"
                aria-expanded={isExpanded}
                aria-controls={contentId}
                tabIndex={0}
                onKeyDown={handleKeyDown}
            >
                <div className="model-card-title">
                    <div onClick={handleSwitchClick}>
                        <Switch checked={isEnabled} onChange={handleEnabledToggle} />
                    </div>
                    <strong>{model.name}</strong>
                </div>
                <div className="expand-indicator">
                    {isEnabled && (
                        <svg
                            width="20"
                            height="20"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            style={{
                                transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
                                transition: 'transform 0.3s',
                            }}
                            aria-hidden="true"
                        >
                            <polyline points="6 9 12 15 18 9" />
                        </svg>
                    )}
                </div>
            </div>

            {/* Collapsed state: Show description and equation ONLY */}
            {!isExpanded && isEnabled && (
                <div className="model-card-collapsed-content">
                    <p className="model-description">{model.shortDescription}</p>
                    <div className="model-equation">
                        <EquationRenderer latex={model.equationLatex} />
                    </div>
                </div>
            )}

            {/* Expanded state: Show configuration form ONLY */}
            {isExpanded && isEnabled && (
                <div className="model-card-content" id={contentId}>
                    <ModelConfigForm
                        modelId={model.id}
                        parameterDefaults={model.parameterDefaults}
                        value={currentConfig}
                        onChange={handleConfigChange}
                    />
                </div>
            )}
        </div>
    );
};

export default ModelCard;
