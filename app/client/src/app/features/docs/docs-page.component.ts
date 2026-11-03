// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

import { Component, DestroyRef, inject } from '@angular/core';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { DOCS_GROUPS } from './docs-content';

@Component({
    selector: 'adsmod-docs-page',
    standalone: true,
    imports: [RouterLink],
    template: `
        <div class="docs-layout">
            <nav class="docs-toc console-card" aria-label="Documentation topics">
                @for (group of groups; track group.id) {
                    <h2>{{ group.label }}</h2>
                    <ul>
                        @for (section of group.sections; track section.id) {
                            <li>
                                <a routerLink="/help" [fragment]="section.id">{{ section.title }}</a>
                            </li>
                        }
                    </ul>
                }
            </nav>
            <article class="docs-content">
                @for (group of groups; track group.id) {
                    @for (section of group.sections; track section.id) {
                        <section class="console-card docs-section" [id]="section.id" [attr.aria-labelledby]="section.id + '-title'">
                            <p class="docs-group">{{ group.label }}</p>
                            <h2 [id]="section.id + '-title'" tabindex="-1">{{ section.title }}</h2>
                            <p class="docs-intro">{{ section.intro }}</p>
                            @for (paragraph of section.paragraphs ?? []; track $index) {
                                <p>{{ paragraph }}</p>
                            }
                            @if (section.steps?.length) {
                                <ol class="docs-steps">
                                    @for (step of section.steps; track step.title) {
                                        <li>
                                            <strong>{{ step.title }}</strong>
                                            <span>{{ step.description }}</span>
                                        </li>
                                    }
                                </ol>
                            }
                            @if (section.tips?.length) {
                                <h3>Tips</h3>
                                <ul class="docs-tips">
                                    @for (tip of section.tips; track $index) {
                                        <li>{{ tip }}</li>
                                    }
                                </ul>
                            }
                        </section>
                    }
                }
            </article>
        </div>
    `,
    styles: [
        `
            :host {
                display: block;
            }
            .docs-layout {
                display: grid;
                grid-template-columns: minmax(13rem, 16rem) minmax(0, 1fr);
                gap: 1rem;
                align-items: start;
                max-width: 1200px;
            }
            .docs-toc {
                position: sticky;
                top: 0;
                max-height: calc(100vh - 8rem);
                overflow: auto;
                padding: 0.9rem 1rem;
            }
            .docs-toc h2 {
                margin: 0.8rem 0 0.3rem;
                color: var(--slate-500);
                font-size: 0.7rem;
                letter-spacing: 0.05em;
                text-transform: uppercase;
            }
            .docs-toc h2:first-child {
                margin-top: 0;
            }
            .docs-toc ul {
                display: grid;
                gap: 0.15rem;
                margin: 0;
                padding: 0;
                list-style: none;
            }
            .docs-toc a {
                display: block;
                padding: 0.25rem 0.4rem;
                border-radius: 6px;
                font-size: 0.85rem;
                text-decoration: none;
            }
            .docs-toc a:hover {
                background: var(--primary-50);
            }
            .docs-content {
                display: grid;
                gap: 1rem;
                min-width: 0;
            }
            .docs-section {
                scroll-margin-top: 0.75rem;
            }
            .docs-group {
                margin: 0;
                color: var(--slate-500);
                font-size: 0.7rem;
                letter-spacing: 0.05em;
                text-transform: uppercase;
            }
            .docs-section h2 {
                margin: 0.15rem 0 0.4rem;
                font-size: 1.25rem;
            }
            .docs-section h3 {
                margin: 1rem 0 0.35rem;
                font-size: 0.9rem;
            }
            .docs-intro {
                color: var(--slate-600, var(--slate-500));
            }
            .docs-section p {
                max-width: 70ch;
                margin: 0 0 0.7rem;
                line-height: 1.55;
            }
            .docs-steps {
                display: grid;
                gap: 0.6rem;
                margin: 0.8rem 0 0;
                padding-left: 1.25rem;
            }
            .docs-steps li {
                max-width: 70ch;
                padding-left: 0.25rem;
            }
            .docs-steps strong {
                display: block;
            }
            .docs-steps span {
                color: var(--slate-600, var(--slate-500));
                font-size: 0.9rem;
            }
            .docs-tips {
                margin: 0;
                padding-left: 1.25rem;
                font-size: 0.9rem;
                line-height: 1.55;
            }
            @media (max-width: 860px) {
                .docs-layout {
                    grid-template-columns: minmax(0, 1fr);
                }
                .docs-toc {
                    position: static;
                    max-height: none;
                }
            }
        `,
    ],
})
export class DocsPageComponent {
    protected readonly groups = DOCS_GROUPS;

    constructor() {
        inject(ActivatedRoute)
            .fragment.pipe(takeUntilDestroyed(inject(DestroyRef)))
            .subscribe((fragment) => {
                if (!fragment) {
                    return;
                }
                // Wait for the sections to render before scrolling to the topic.
                window.setTimeout(() => {
                    const target = document.getElementById(fragment);
                    target?.scrollIntoView({ block: 'start' });
                    target?.querySelector<HTMLElement>('h2')?.focus({ preventScroll: true });
                });
            });
    }
}
