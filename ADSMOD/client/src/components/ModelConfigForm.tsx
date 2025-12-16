import React, { useCallback } from 'react';
import { NumberInput } from './UIComponents';
import type { ModelParameters } from '../types';

interface ModelConfigFormProps {
    modelId: string;
    parameterDefaults: Record<string, [number, number]>;
    value: ModelParameters;
    onChange: (value: ModelParameters) => void;
}

/**
 * ModelConfigForm: Renders parameter configuration inputs for a model.
 * Generates fields based on parameter defaults and displays min/max bounds inputs.
 */
export const ModelConfigForm: React.FC<ModelConfigFormProps> = ({
    modelId,
    parameterDefaults,
    value,
    onChange,
}) => {
    const handleParameterChange = useCallback(
        (paramName: string, boundType: 'min' | 'max', newValue: number) => {
            const updated = {
                ...value,
                [paramName]: {
                    ...value[paramName],
                    [boundType]: newValue,
                },
            };
            onChange(updated);
        },
        [value, onChange]
    );

    const parameterEntries = Object.entries(parameterDefaults);

    if (parameterEntries.length === 0) {
        return (
            <div className="config-form-empty">
                <p>Configuration is not available for this model.</p>
            </div>
        );
    }

    return (
        <div className="model-config-form" id={`config-form-${modelId}`}>
            <div className="config-form-fields">
                {parameterEntries.map(([paramName, [minDefault, maxDefault]]) => (
                    <div key={paramName} className="parameter-row">
                        <div className="parameter-label">{paramName}</div>
                        <div className="parameter-inputs">
                            <NumberInput
                                label="min"
                                value={value[paramName]?.min ?? minDefault}
                                onChange={(val) => handleParameterChange(paramName, 'min', val)}
                                min={0}
                                step={0.0001}
                                precision={4}
                            />
                            <NumberInput
                                label="max"
                                value={value[paramName]?.max ?? maxDefault}
                                onChange={(val) => handleParameterChange(paramName, 'max', val)}
                                min={0}
                                step={0.0001}
                                precision={4}
                            />
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default ModelConfigForm;
