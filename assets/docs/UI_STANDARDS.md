# ADSMOD UI Standards

Last updated: 2026-04-24

This standard is based on the current implementation in `ADSMOD/client/src`.

## 1. Typography

- Primary font stack: `'Space Grotesk', 'Inter', 'Segoe UI', sans-serif`.
- Monospace stack for logs/code: `'JetBrains Mono', 'Fira Code', ...`.
- Typical scale in use:
  - Page/brand headings: ~`1.35rem` to `2rem`
  - Section titles: ~`1.05rem` to `1.25rem`
  - Body text: ~`0.875rem` to `0.95rem`
  - Micro/meta labels: ~`0.7rem` to `0.85rem`
- Maintain readable line heights (generally `1.4` to `1.6`) for dense technical content.

## 2. Layout and Spacing

- Tokenized spacing system (`index.css`):
  - `--spacing-xs: 0.5rem`
  - `--spacing-sm: 0.75rem`
  - `--spacing-md: 1rem`
  - `--spacing-lg: 1.5rem`
  - `--spacing-xl: 2rem`
- Radius tokens:
  - `--radius-sm: 0.5rem`
  - `--radius-md: 0.75rem`
  - `--radius-lg: 1rem`
  - `--radius-xl: 1.5rem`
- Layout patterns:
  - Top sticky header + tab navigation.
  - `source` page: two-column panel layout.
  - `fitting` page: controls + status/log column, model card grid.
  - `training` page: left view toolbar + right active workspace panel.

## 3. Color System

### Core palettes

- Neutral/slate scale: `--slate-50` through `--slate-900`.
- Primary blue scale: `--primary-50` through `--primary-700`.

### Semantic usage (current implementation)

- Success: green families (for example `#dcfce7`, `#166534`, `#86efac`).
- Error/destructive: red families (for example `#ef4444`, `#dc2626`, `#fee2e2`).
- Warning: amber families (for example `#fef3c7`, `#92400e`).
- Info: blue-tinted status surfaces (for example `#e0f2fe`, primary blue tokens).

### Surface usage

- App background: light slate tint (`#eef2f7`, `#f6f8fc`).
- Primary content surfaces: white or light gradients.
- Logs/code panes: dark slate background with light text.

## 4. Components and Interaction Patterns

- Buttons:
  - `button.primary`: blue gradient call-to-action.
  - `button.secondary`: bordered neutral action.
  - `ghost-button`: low-emphasis utility action.
  - Disabled state uses opacity and non-interactive cursor.
- Forms:
  - `.select-input`, numeric/text inputs with hover/focus border changes.
  - Consistent rounded corners via radius tokens.
- Navigation:
  - Header tabs (`.header-tab`) with active and hover states.
  - Training sub-navigation tabs (`.training-view-tab`) with active state.
- Data views:
  - Card and panel patterns for sections.
  - Scrollable log and markdown regions for long output.
- Modals:
  - Backdrop overlay + centered dialog containers (`.modal-backdrop` and related modal components).

## 5. Page Structure and Navigation Hierarchy

- Global top-level routes/pages:
  - `source`
  - `fitting`
  - `training`
- `training` contains sub-views:
  - `Data Processing`
  - `Train datasets`
  - `Checkpoints`
  - `Training Dashboard`
- Navigation is state-driven in `App.tsx` and `MachineLearningPage.tsx`.

## 6. User Experience Rules

- Core workflows are long-running and status-heavy; always surface:
  - explicit start/running/success/error/cancel states,
  - progress indicators and logs,
  - actionable error messages.
- Keep behavior consistent across dataset ingestion, fitting, NIST collection, and training flows.
- Maintain clear empty states (for example, no datasets/checkpoints available).
- Preserve non-blocking UX by polling backend job endpoints for progress.

## 7. Responsiveness

- Current breakpoints include (and are not limited to):
  - `1480px`, `1360px`, `1320px`, `1200px`, `1180px`, `1100px`, `1080px`, `900px`, `768px`, `760px`, `720px`, `700px`, `600px`.
- Required responsive behavior:
  - Collapse multi-column layouts to single-column on narrower widths.
  - Convert training toolbar from vertical to wrapped horizontal layout at reduced widths.
  - Reduce model grids progressively to 3/2/1 columns based on viewport.

## 8. Accessibility

- Keyboard support is required for interactive elements, including custom row/button-like elements (`tabIndex`, `onKeyDown` patterns already in use).
- Preserve ARIA patterns currently implemented:
  - `aria-label` for navigation and control buttons.
  - `role="dialog"` + modal ARIA attributes for modal overlays.
  - `aria-expanded`/`aria-controls` for expandable cards.
- Focus visibility:
  - Keep explicit `:focus-visible` outlines on controls.
- Motion accessibility:
  - Respect `prefers-reduced-motion` overrides already defined in CSS.

## 9. Design Principles

- Consistency over one-off styling.
- Clarity and predictability for technical workflows.
- Visual complexity only where it improves comprehension.
- Keep token-driven styling; avoid introducing hardcoded, conflicting style systems.
