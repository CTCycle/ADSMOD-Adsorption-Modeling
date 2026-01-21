import React, { useState, useRef } from 'react';

interface ExpansionProps {
    title: string;
    children: React.ReactNode;
    disabled?: boolean;
    defaultOpen?: boolean;
}

export const Expansion: React.FC<ExpansionProps> = ({ title, children, disabled = false, defaultOpen = false }) => {
    const [isOpen, setIsOpen] = useState(defaultOpen);

    return (
        <div className={`expansion ${disabled ? 'disabled' : ''}`}>
            <div className="expansion-header" onClick={() => !disabled && setIsOpen(!isOpen)}>
                <span>{title}</span>
                <span style={{ transform: isOpen ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.3s' }}>
                    ▼
                </span>
            </div>
            {isOpen && <div className="expansion-content">{children}</div>}
        </div>
    );
};

interface SwitchProps {
    checked: boolean;
    onChange: (checked: boolean) => void;
    label?: string;
}

export const Switch: React.FC<SwitchProps> = ({ checked, onChange, label }) => {
    return (
        <div className="switch-container">
            <label className="switch">
                <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
                <span className="slider"></span>
            </label>
            {label && <span>{label}</span>}
        </div>
    );
};

interface NumberInputProps {
    label: string;
    value: number;
    onChange: (value: number) => void;
    min?: number;
    max?: number;
    step?: number;
    precision?: number;
}

export const NumberInput: React.FC<NumberInputProps> = ({
    label,
    value,
    onChange,
    min,
    max,
    step = 0.0001,
    precision = 4,
}) => {
    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const newValue = parseFloat(e.target.value);
        if (!isNaN(newValue)) {
            onChange(Number(newValue.toFixed(precision)));
        }
    };

    return (
        <div style={{ flex: 1, minWidth: '140px' }}>
            <label>{label}</label>
            <input
                type="number"
                value={value}
                onChange={handleChange}
                min={min}
                max={max}
                step={step}
            />
        </div>
    );
};

interface CheckboxProps {
    label: string;
    checked: boolean;
    onChange: (checked: boolean) => void;
}

export const Checkbox: React.FC<CheckboxProps> = ({ label, checked, onChange }) => {
    return (
        <div className="checkbox-container">
            <input
                type="checkbox"
                checked={checked}
                onChange={(e) => onChange(e.target.checked)}
                id={`checkbox-${label}`}
            />
            <label htmlFor={`checkbox-${label}`} style={{ marginBottom: 0, cursor: 'pointer' }}>
                {label}
            </label>
        </div>
    );
};

interface FileUploadProps {
    label: string;
    accept: string;
    onSelect?: (file: File) => void;
    onUpload?: (file: File) => void;
    autoUpload?: boolean;
    disabled?: boolean;
}

export const FileUpload: React.FC<FileUploadProps> = ({
    label,
    accept,
    onSelect,
    onUpload,
    autoUpload = true,
    disabled = false,
}) => {
    const inputRef = useRef<HTMLInputElement>(null);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            if (onSelect) {
                onSelect(file);
            }
            if (autoUpload && onUpload) {
                onUpload(file);
            }
        }
        // Reset the input value so the same file can be selected again
        if (inputRef.current) {
            inputRef.current.value = '';
        }
    };

    return (
        <div className="file-upload">
            <input ref={inputRef} type="file" accept={accept} onChange={handleChange} disabled={disabled} />
            <div className="file-upload-label">
                <span>📁</span>
                <span>{label}</span>
            </div>
        </div>
    );
};
