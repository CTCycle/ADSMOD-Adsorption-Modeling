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
                padding: 1.5rem 1.75rem 1.25rem;
                scroll-margin-top: 0.75rem;
            }
            .docs-group {
                margin: 0;
                color: var(--slate-500);
                font-size: 0.7rem;
                font-weight: 600;
                letter-spacing: 0.06em;
                text-transform: uppercase;
            }
            .docs-section h2 {
                margin: 0.25rem 0 0.6rem;
                font-size: 1.4rem;
                line-height: 1.25;
            }
            .docs-section h3 {
                margin: 1.25rem 0 0.4rem;
                font-size: 0.8rem;
                letter-spacing: 0.05em;
                text-transform: uppercase;
                color: var(--slate-500);
            }
            .docs-section p {
                max-width: 68ch;
                margin: 0 0 0.75rem;
                font-size: 0.95rem;
                line-height: 1.65;
            }
            .docs-section .docs-intro {
                font-size: 1.02rem;
                color: var(--slate-600, var(--slate-500));
            }
            .docs-steps {
                display: grid;
                gap: 0.9rem;
                margin: 1rem 0 0;
                padding: 0;
                list-style: none;
                counter-reset: step;
            }
            .docs-steps li {
                position: relative;
                max-width: 68ch;
                padding-left: 2.5rem;
                counter-increment: step;
            }
            .docs-steps li::before {
                content: counter(step);
                position: absolute;
                left: 0;
                top: 0.05rem;
                display: grid;
                place-items: center;
                width: 1.7rem;
                height: 1.7rem;
                border-radius: 50%;
                background: var(--primary-50);
                color: var(--primary-700, var(--primary-600, inherit));
                font-size: 0.8rem;
                font-weight: 600;
            }
            .docs-steps strong {
                display: block;
                font-size: 0.95rem;
                line-height: 1.4;
            }
            .docs-steps span {
                display: block;
                margin-top: 0.15rem;
                color: var(--slate-600, var(--slate-500));
                font-size: 0.9rem;
                line-height: 1.6;
            }
            .docs-tips {
                max-width: 68ch;
                margin: 0;
                padding: 0.7rem 1rem 0.7rem 2rem;
                border-left: 3px solid var(--primary-200, var(--primary-50));
                border-radius: 0 8px 8px 0;
                background: var(--primary-50);
                font-size: 0.9rem;
                line-height: 1.6;
            }
            .docs-tips li + li {
                margin-top: 0.3rem;
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
