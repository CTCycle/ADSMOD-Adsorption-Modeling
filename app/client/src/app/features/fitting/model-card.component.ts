import { Component, EventEmitter, Input, Output } from '@angular/core';
import type { AdsorptionModel } from '../../models/adsorption-model.model';
import type { ModelParameters } from '../../models/fitting.model';
import { EquationRendererComponent } from '../../shared/components/equation-renderer/equation-renderer.component';
import { SwitchComponent } from '../../shared/components/switch/switch.component';
import { ModelConfigFormComponent } from './model-config-form.component';

@Component({
    selector: 'adsmod-model-card',
    standalone: true,
    imports: [EquationRendererComponent, ModelConfigFormComponent, SwitchComponent],
    template: `
        <div
            class="model-grid-card"
            [class.expanded]="isExpanded"
            [class.disabled]="!isEnabled"
            [id]="cardId"
        >
            <div
                class="model-card-header"
                role="button"
                [attr.aria-expanded]="isExpanded"
                [attr.aria-controls]="contentId"
                tabindex="0"
                (click)="handleHeaderClick()"
                (keydown)="handleKeyDown($event)"
            >
                <div class="model-card-title">
                    <div (click)="$event.stopPropagation()">
                        <adsmod-switch [checked]="isEnabled" (checkedChange)="enabledChange.emit($event)" />
                    </div>
                    <strong>{{ model.name }}</strong>
                </div>
                <div class="expand-indicator">
                    @if (isEnabled) {
                        <svg
                            width="20"
                            height="20"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            stroke-width="2"
                            [style.transform]="isExpanded ? 'rotate(180deg)' : 'rotate(0deg)'"
                            style="transition: transform 0.3s;"
                            aria-hidden="true"
                        >
                            <polyline points="6 9 12 15 18 9" />
                        </svg>
                    }
                </div>
            </div>

            @if (!isExpanded && isEnabled) {
                <div class="model-card-collapsed-content">
                    <p class="model-description">{{ model.shortDescription }}</p>
                    <div class="model-equation">
                        <adsmod-equation-renderer [latex]="model.equationLatex" />
                    </div>
                </div>
            }

            @if (isExpanded && isEnabled) {
                <div class="model-card-content" [id]="contentId">
                    <adsmod-model-config-form
                        [modelId]="model.id"
                        [parameterDefaults]="model.parameterDefaults"
                        [value]="currentConfig"
                        (valueChange)="configChange.emit($event)"
                    />
                </div>
            }
        </div>
    `,
})
export class ModelCardComponent {
    @Input({ required: true }) model!: AdsorptionModel;
    @Input() isExpanded = false;
    @Input() isEnabled = true;
    @Input({ required: true }) currentConfig: ModelParameters = {};
    @Output() readonly toggle = new EventEmitter<string>();
    @Output() readonly enabledChange = new EventEmitter<boolean>();
    @Output() readonly configChange = new EventEmitter<ModelParameters>();

    protected get cardId(): string {
        return `model-card-${this.model.id}`;
    }

    protected get contentId(): string {
        return `model-content-${this.model.id}`;
    }

    protected handleHeaderClick(): void {
        if (this.isEnabled) {
            this.toggle.emit(this.model.id);
        }
    }

    protected handleKeyDown(event: KeyboardEvent): void {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            this.handleHeaderClick();
        }
    }
}
