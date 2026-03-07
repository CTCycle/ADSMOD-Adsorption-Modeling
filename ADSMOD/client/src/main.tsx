import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import './wizard-styles.css';
import { initializeRuntimeConfig } from './runtimeConfig';

async function bootstrap(): Promise<void> {
    await initializeRuntimeConfig();
    const { default: App } = await import('./App.tsx');

    createRoot(document.getElementById('root')!).render(
        <StrictMode>
            <App />
        </StrictMode>
    );
}

void bootstrap();
