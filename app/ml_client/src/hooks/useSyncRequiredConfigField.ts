import { useEffect } from 'react';

export const useSyncRequiredConfigField = <
    TConfig extends object,
    TKey extends keyof TConfig,
>(
    config: TConfig,
    field: TKey,
    requiredValue: TConfig[TKey],
    onConfigChange: (nextConfig: TConfig) => void
): void => {
    useEffect(() => {
        if (config[field] !== requiredValue) {
            onConfigChange({ ...config, [field]: requiredValue });
        }
    }, [config, field, requiredValue, onConfigChange]);
};
