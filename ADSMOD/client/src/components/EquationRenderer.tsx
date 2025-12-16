import React from 'react';

interface EquationRendererProps {
    latex: string;
    displayMode?: boolean;
}

/**
 * EquationRenderer: Renders LaTeX equations using KaTeX.
 * Falls back to displaying raw LaTeX if rendering fails.
 */
export const EquationRenderer: React.FC<EquationRendererProps> = ({ latex, displayMode = true }) => {
    const [html, setHtml] = React.useState<string>('');
    const [error, setError] = React.useState<boolean>(false);

    React.useEffect(() => {
        let isMounted = true;

        const renderEquation = async () => {
            try {
                // Dynamically import KaTeX
                const katex = await import('katex');
                if (isMounted) {
                    const rendered = katex.default.renderToString(latex, {
                        throwOnError: false,
                        displayMode: displayMode,
                        output: 'html',
                    });
                    setHtml(rendered);
                    setError(false);
                }
            } catch (err) {
                console.error('KaTeX rendering error:', err);
                if (isMounted) {
                    setError(true);
                }
            }
        };

        renderEquation();

        return () => {
            isMounted = false;
        };
    }, [latex, displayMode]);

    if (error || !html) {
        return (
            <div className="equation-fallback">
                <code>{latex}</code>
            </div>
        );
    }

    return (
        <div
            className="equation-container"
            dangerouslySetInnerHTML={{ __html: html }}
        />
    );
};

export default EquationRenderer;
