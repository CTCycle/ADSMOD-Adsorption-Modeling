# Architecture Overview

Last updated: 2026-06-03

## Scope

This section covers system structure, ownership boundaries, API routing, and backend workspace organization.

## Documents

- [`system_overview.md`](system_overview.md)
  - Repository layout, service split, frontend split, and primary entry points.
- [`service_boundaries.md`](service_boundaries.md)
  - Allowed dependency directions and prohibited imports.
- [`api_surface.md`](api_surface.md)
  - Core and ML route ownership.
- [`persistence_and_packages.md`](persistence_and_packages.md)
  - Shared workspace layout, persistence ownership, and architecture validation expectations.

## When To Read Which File

- Read [`system_overview.md`](system_overview.md) for orientation and directory layout.
- Read [`service_boundaries.md`](service_boundaries.md) before changing package imports or moving backend code.
- Read [`api_surface.md`](api_surface.md) before adding, relocating, or reviewing endpoints.
- Read [`persistence_and_packages.md`](persistence_and_packages.md) before changing repositories, models, sessions, or backend packaging.
