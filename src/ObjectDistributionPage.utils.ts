import {
  type ObjectDistributionPoint,
  type ObjectDistributionResult,
} from './pose-trace/types';
import { type H5File } from './stores';

export type ObjectViewTab = 'scatter' | 'position' | 'rotation' | 'slices';

export type DatasetRole = 'success' | 'failed' | 'teleop';

export interface DatasetGroupSelection {
  successUrl: string | null;
  failedUrl: string | null;
  teleopUrl: string | null;
}

export interface PlotClickEvent {
  points?: {
    pointIndex?: number;
    data?: { name?: string };
  }[];
}

// File-name convention for a dataset pack (all three share the same base name):
//   <name>_annotated_<string>          → teleop
//   <name>_generated_<string>          → successful generated
//   <name>_generated_<string>_failed   → failed generated
// `failed` is checked before `success` because a failed file is also generated.
export function classifyDataset(file: H5File): DatasetRole | null {
  const name = file.name.toLowerCase();
  const isGenerated = name.includes('generated');
  if (isGenerated && name.includes('failed')) {
    return 'failed';
  }
  if (isGenerated) {
    return 'success';
  }
  if (name.includes('annotated') || name.includes('teleop')) {
    return 'teleop';
  }

  return null;
}

// The shared prefix that ties a pack together: everything before the
// `_generated` / `_annotated` role marker (extension stripped).
export function datasetBaseName(file: H5File): string {
  const stem = file.name.toLowerCase().replace(/\.[^./\\]+$/u, '');
  return /^(.*?)_(?:annotated|generated)/u.exec(stem)?.[1] ?? stem;
}

// Pick a coherent pack of (success, failed, teleop) datasets. Files are grouped
// by their shared base name so opening several packs never mixes roles across
// packs; within the chosen pack each role is assigned from the name convention.
export function selectDatasetGroup(
  files: H5File[],
  preferredUrl: string | null,
): DatasetGroupSelection {
  const groups = new Map<string, H5File[]>();
  for (const file of files) {
    const base = datasetBaseName(file);
    groups.set(base, [...(groups.get(base) ?? []), file]);
  }

  let targetBase: string | null = null;
  const preferred = preferredUrl
    ? files.find((file) => file.url === preferredUrl)
    : null;
  if (preferred) {
    targetBase = datasetBaseName(preferred);
  } else {
    // Prefer the pack with the most distinct roles present (a complete 3-file
    // pack wins), keeping the first such pack on ties.
    let bestScore = -1;
    for (const [base, packFiles] of groups) {
      const score = new Set(packFiles.map(classifyDataset).filter(Boolean))
        .size;
      if (score > bestScore) {
        bestScore = score;
        targetBase = base;
      }
    }
  }

  const packFiles =
    targetBase === null ? files : (groups.get(targetBase) ?? []);
  function pick(role: DatasetRole): string | null {
    return (
      packFiles.find((file) => classifyDataset(file) === role)?.url ?? null
    );
  }

  return {
    successUrl: pick('success'),
    failedUrl: pick('failed'),
    teleopUrl: pick('teleop'),
  };
}

export function analysisTitle(tab: ObjectViewTab): string {
  if (tab === 'position') {
    return 'Position Failure Map';
  }

  if (tab === 'rotation') {
    return 'Rotation Failure Map';
  }

  return 'Spatial Failure Slices';
}

export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

export function formatSupport(value: number): string {
  return value.toFixed(2);
}

export function formatRange(start: number, end: number, unit: string): string {
  return `${start.toFixed(2)} to ${end.toFixed(2)} ${unit}`;
}

export function hashRevisionKey(parts: (string | number | boolean)[]): number {
  const key = parts.join(':');
  let hash = 0;
  for (let index = 0; index < key.length; index += 1) {
    hash = (hash * 31 + key.charCodeAt(index)) % 4_294_967_296;
  }

  return hash;
}

export function resolveClickedPoint(
  event: PlotClickEvent,
  result: ObjectDistributionResult | null,
): ObjectDistributionPoint | null {
  const point = event.points?.[0];
  const pointIndex = point?.pointIndex ?? -1;
  if (!point || pointIndex < 0 || !result) {
    return null;
  }

  if (point.data?.name === 'Success') {
    return result.successPoints[pointIndex] ?? null;
  }

  if (point.data?.name === 'Failed') {
    return result.failedPoints[pointIndex] ?? null;
  }

  return null;
}
