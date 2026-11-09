// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, DestroyRef, ElementRef, effect, inject, input, output, signal, viewChild } from '@angular/core';
import type { GLViewer } from '3dmol';

export type StructureFormat = 'sdf' | 'cif';
type RenderStyle = 'ball-stick' | 'stick' | 'sphere';

/**
 * 3Dmol only expands symmetry from the legacy `_symmetry_equiv_pos_*` tags, while current COD files use
 * `_space_group_symop_*`; renaming the tags lets it generate the full unit cell instead of one asymmetric unit.
 */
function withLegacySymmetryTags(cif: string): string {
    return cif
        .replaceAll('_space_group_symop_operation_xyz', '_symmetry_equiv_pos_as_xyz')
        .replaceAll('_space_group_symop_id', '_symmetry_equiv_pos_site_id');
}

/** Expanded, interactive 3D view; the 3Dmol library loads only when a structure is first opened. */
@Component({
    selector: 'adsmod-structure-viewer-dialog',
    standalone: true,
    template: `
        <div class="viewer-backdrop" (click)="closed.emit()">
            <section
                #dialog
                class="viewer-dialog"
                role="dialog"
                aria-modal="true"
                aria-labelledby="structure-viewer-title"
                tabindex="-1"
                (click)="$event.stopPropagation()"
                (keydown.escape)="$event.stopPropagation(); closed.emit()"
            >
                <header>
                    <div>
                        <h2 id="structure-viewer-title">{{ title() }}</h2>
                        @if (subtitle()) {
                            <p>{{ subtitle() }}</p>
                        }
                    </div>
                    <button class="button quiet" type="button" aria-label="Close 3D viewer" (click)="closed.emit()">✕</button>
                </header>
                <div class="viewer-stage">
                    <div #host class="viewer-host" role="img" [attr.aria-label]="'Interactive 3D view of ' + title()"></div>
                    @if (status(); as message) {
                        <p class="viewer-status" role="status">{{ message }}</p>
                    }
                </div>
                <footer>
                    <label>
                        Style
                        <select [value]="style()" (change)="setStyle($any($event.target).value)">
                            <option value="ball-stick">Ball and stick</option>
                            <option value="stick">Sticks</option>
                            <option value="sphere">Spheres</option>
                        </select>
                    </label>
                    <span>Drag to rotate · scroll to zoom</span>
                </footer>
            </section>
        </div>
    `,
    styles: [
        `
            .viewer-backdrop { position: fixed; inset: 0; z-index: 60; display: grid; place-items: center; padding: 1rem; background: rgb(3 24 47 / 0.45); }
            .viewer-dialog { display: grid; grid-template-rows: auto minmax(0, 1fr) auto; width: min(52rem, 100%); height: min(38rem, 100%); border: 1px solid var(--border-color); border-radius: 8px; background: var(--surface); box-shadow: 0 24px 60px rgb(3 24 47 / 0.35); }
            .viewer-dialog:focus { outline: none; }
            header { display: flex; align-items: flex-start; justify-content: space-between; gap: 0.75rem; padding: 0.8rem 1rem; border-bottom: 1px solid var(--border-color); }
            header h2 { margin: 0; font-size: 1rem; overflow-wrap: anywhere; }
            header p { margin: 2px 0 0; color: var(--text-secondary); font-size: 0.78rem; }
            .viewer-stage { position: relative; min-height: 0; background: #fff; }
            .viewer-host { position: absolute; inset: 0; }
            .viewer-status { position: absolute; inset: 0; display: grid; place-items: center; margin: 0; padding: 1rem; background: var(--surface); color: var(--text-secondary); font-size: 0.85rem; text-align: center; }
            footer { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 0.5rem; padding: 0.55rem 1rem; border-top: 1px solid var(--border-color); color: var(--text-secondary); font-size: 0.78rem; }
            footer label { display: inline-flex; align-items: center; gap: 0.4rem; }
        `,
    ],
})
export class StructureViewerDialogComponent {
    readonly title = input.required<string>();
    readonly subtitle = input<string | null>(null);
    readonly format = input.required<StructureFormat>();
    readonly content = input.required<string>();
    readonly closed = output<void>();

    private readonly host = viewChild.required<ElementRef<HTMLElement>>('host');
    private readonly dialog = viewChild.required<ElementRef<HTMLElement>>('dialog');
    protected readonly style = signal<RenderStyle>('ball-stick');
    protected readonly status = signal<string | null>('Loading 3D viewer…');
    private viewer: GLViewer | null = null;
    private viewerHost: HTMLElement | null = null;
    private destroyed = false;

    constructor() {
        inject(DestroyRef).onDestroy(() => this.releaseViewer());
        effect(() => {
            this.dialog().nativeElement.focus();
            const format = this.format();
            const content = this.content();
            void this.render(format, content);
        });
    }

    protected setStyle(style: RenderStyle): void {
        this.style.set(style);
        this.applyStyle();
    }

    private async render(format: StructureFormat, content: string): Promise<void> {
        try {
            // 3Dmol is CommonJS: the bundler exposes its API under `default`.
            const library = (await import('3dmol')).default;
            if (this.destroyed) return;
            this.viewer?.clear();
            if (!this.viewer) {
                this.viewerHost = this.host().nativeElement;
                this.viewer = library.createViewer(this.viewerHost, { backgroundColor: 'white' });
            }
            const options = format === 'cif' ? { doAssembly: true, normalizeAssembly: true, duplicateAssemblyAtoms: true } : {};
            this.viewer.addModel(format === 'cif' ? withLegacySymmetryTags(content) : content, format, options);
            this.applyStyle();
            if (format === 'cif') {
                this.viewer.addUnitCell(this.viewer.getModel());
            }
            this.viewer.zoomTo();
            this.viewer.render();
            this.status.set(null);
        } catch {
            this.status.set('This structure could not be displayed.');
        }
    }

    /** Free the WebGL context eagerly: browsers cap live contexts and clear() alone keeps the canvas. */
    private releaseViewer(): void {
        this.destroyed = true;
        this.viewer?.clear();
        const canvas = this.viewerHost?.querySelector('canvas');
        const context = canvas?.getContext('webgl2') ?? canvas?.getContext('webgl');
        context?.getExtension('WEBGL_lose_context')?.loseContext();
        this.viewerHost?.replaceChildren();
        this.viewer = null;
        this.viewerHost = null;
    }

    private applyStyle(): void {
        const viewer = this.viewer;
        if (!viewer) return;
        const style = this.style();
        viewer.setStyle(
            {},
            style === 'sphere'
                ? { sphere: { scale: 0.6, colorscheme: 'Jmol' } }
                : style === 'stick'
                  ? { stick: { radius: 0.15, colorscheme: 'Jmol' } }
                  : { stick: { radius: 0.12, colorscheme: 'Jmol' }, sphere: { scale: 0.25, colorscheme: 'Jmol' } },
        );
        viewer.render();
    }
}
