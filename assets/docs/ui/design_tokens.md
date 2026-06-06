# ADSMOD Design Tokens

Last updated: 2026-06-03

## Typography

- Primary font stack: `'Space Grotesk', 'Inter', 'Segoe UI', sans-serif`
- Monospace stack for logs and code: `'JetBrains Mono', 'Fira Code', ...`
- Typical scale in use
  - page and brand headings: about `1.35rem` to `2rem`
  - section titles: about `1.05rem` to `1.25rem`
  - body text: about `0.875rem` to `0.95rem`
  - micro and meta labels: about `0.7rem` to `0.85rem`
- Maintain readable line heights, generally `1.4` to `1.6`, for dense technical content.

## Spacing And Radius

- Spacing tokens from `index.css`
  - `--spacing-xs: 0.5rem`
  - `--spacing-sm: 0.75rem`
  - `--spacing-md: 1rem`
  - `--spacing-lg: 1.5rem`
  - `--spacing-xl: 2rem`
- Radius tokens
  - `--radius-sm: 0.5rem`
  - `--radius-md: 0.75rem`
  - `--radius-lg: 1rem`
  - `--radius-xl: 1.5rem`

## Color System

- Neutral slate scale: `--slate-50` through `--slate-900`
- Primary blue scale: `--primary-50` through `--primary-700`
- Semantic usage
  - success uses green families
  - error and destructive states use red families
  - warning states use amber families
  - informational states use blue-tinted surfaces

## Surface Usage

- App background uses light slate tints such as `#eef2f7` and `#f6f8fc`
- Primary content surfaces use white or light gradients
- Logs and code panes use dark slate backgrounds with light text
