# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

myHDF5 is a free online service to explore and visualize HDF5 files. Built with React 18, TypeScript, and Vite. Uses h5wasm (WebAssembly) for client-side HDF5 reading and @h5web/app for visualization. Deployed to Netlify from `main`.

## Commands

```bash
pnpm install                    # Install dependencies
pnpm start                      # Dev server (auto-opens browser)
pnpm build                      # Production build
pnpm preview                    # Serve production build locally

pnpm lint                       # Run all checks (eslint + tsc + prettier)
pnpm lint:eslint                # ESLint only (--fix to auto-fix)
pnpm lint:tsc                   # TypeScript type checking
pnpm lint:prettier              # Prettier check (--write to format)
pnpm analyze                    # Bundle size analysis (run after build)
```

No test runner — code quality is enforced via ESLint (zero warnings policy), TypeScript strict mode, and Prettier.

## Architecture

**Stack:** React 18 + TypeScript + Vite 7 + pnpm (Node 24.x required). SWC for transpilation. ES2020 target for BigInt support (needed by h5wasm).

**Key pages (React Router):**
- `/` — File source selection (local upload or remote URL)
- `/view` — HDF5 file viewer (main feature, file URL passed as query param)
- `/pose-trace` — Pose trace 3D/2D visualization with Plotly
- `/dataset-processing` — Dataset manipulation tools
- `/video-converter` — Video conversion tool

**State:** Zustand store (`stores.ts`) manages opened files. Persisted to localStorage, but local file blobs are excluded from persistence.

**File handling flow:** Files enter via drag-and-drop (react-dropzone) or URL. Remote URLs are resolved through service adapters in `src/services/` that handle GitHub, GitLab, Zenodo, and direct URLs. `ViewerContainer` resolves files and routes to `LocalFileViewer` or `RemoteFileViewer`.

**Feature modules:**
- `src/pose-trace/` — Plotly-based pose visualization with web worker for heavy computation
- `src/services/` — Platform-specific URL resolution (GitHub raw URLs, GitLab API, Zenodo)
- `src/sidebar/` — File management sidebar with flyout menu

## Conventions

- CSS Modules for component styling (`*.module.css` imported as `styles`)
- Functional components with hooks throughout
- Suspense + suspend-react for async data loading
- Error boundaries via react-error-boundary
- ESLint config from `@esrf/eslint-config` (flat config format)
- Exact dependency versions in package.json (no caret/tilde prefixes for direct deps)
- `.so` files are included as static assets (HDF5 compression plugins)
