# ADSMOD UI standards

Last updated: 2026-07-17

These standards refine the current console UI without changing its information architecture.

## Spacing scale

Use the existing rem-based rhythm: `0.5rem`, `0.75rem`, `1rem`, `1.5rem`, and `2rem` (`--spacing-xs` through `--spacing-xl`). Prefer these tokens for gaps, padding, and section separation. A one-off value is acceptable only when it is required for an icon, control geometry, or a data-table column.

## Typography scale

- Page title: `2rem`, weight 700.
- Section title: `1.25rem`, weight 600.
- Card title: `1.125rem`, weight 600.
- Body/control text: `0.875rem` to `0.95rem`, line-height 1.5.
- Metadata/eyebrow: `0.75rem` to `0.8125rem`, weight 600–700.

Use the existing Space Grotesk stack for interface text and a monospace stack only for logs, code, and tabular technical output.

## Color system

- Slate tokens provide text, borders, and neutral surfaces.
- Primary blue tokens are reserved for navigation, primary actions, focus, and progress.
- Green, amber, and red are semantic status colors and must not be the only indication of state.
- Body text should use `--slate-700` or darker on light surfaces; muted text should remain readable at normal size.

## Component usage rules

- Use `.primary` for the one main action in a section and `.secondary` for reversible or supporting actions.
- Keep controls at least 42px high where practical; icon-only controls need an accessible name and a visible focus state.
- Label every form control. Spreadsheet cells use an `aria-label` combining column and row context.
- Use live regions for asynchronous status messages and `role="progressbar"` for determinate progress.
- Put wide tables inside an explicit scroll container on narrow screens; do not shrink technical columns until their values become unreadable.
- Keep dialog title, `aria-modal`, Escape behavior, and focus management aligned.

## Do and Don't

| Do | Don't |
| --- | --- |
| Reuse spacing, color, radius, and shadow tokens. | Add another near-duplicate blue, gray, or radius for one component. |
| Preserve existing page structure while improving states and alignment. | Introduce a new navigation model or visual theme during polish work. |
| Prefer named classes for static presentation. | Add inline layout/style declarations to templates. |
| Pair color with text, icon, or state copy. | Use a colored dot as the sole status explanation. |
| Validate desktop and narrow layouts plus keyboard focus. | Treat a successful TypeScript build as proof of visual correctness. |

