// Copyright © 2023 Thomas Virdis
// Licensed under the MIT License.

export type JsonPrimitive = string | number | boolean | null;
export type JsonValue = JsonPrimitive | JsonObject | JsonValue[];

export interface JsonObject {
    [key: string]: JsonValue;
}

export type InfoModalValue = JsonValue | undefined;
export type InfoModalData = Record<string, InfoModalValue>;
