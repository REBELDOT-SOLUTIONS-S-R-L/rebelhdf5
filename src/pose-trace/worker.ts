import h5wasm, {
  type Dataset as H5WasmDataset,
  type Entity,
  type File as H5WasmFile,
  type Group as H5WasmGroup,
} from 'h5wasm';
import { Plugin } from '@h5web/h5wasm';

import { getPlugin } from '../plugin-utils';
import {
  TRACE_EEF_KEYPOINT_GROUPS,
  TRACE_EEF_NAMES,
  TRACE_KEYPOINT_NAMES,
  TRACE_SUCCESS_DISTANCE_SPECS,
} from './schema';
import type { DemoInfo, DemoRow } from './types';

type PoseSeries = number[][][];
type H5Module = Awaited<typeof h5wasm.ready>;

type OpenLocalSourcePayload = {
  file: File;
};

type OpenRemoteSourcePayload = {
  buffer: ArrayBuffer;
  name: string;
};

type LoadDemoRowsPayload = {
  sourceId: string;
  demoName: string;
};

type CloseSourcePayload = {
  sourceId: string;
};

type PoseTraceWorkerRequest =
  | { id: number; type: 'openLocalSource'; payload: OpenLocalSourcePayload }
  | { id: number; type: 'openRemoteSource'; payload: OpenRemoteSourcePayload }
  | { id: number; type: 'loadDemoRows'; payload: LoadDemoRowsPayload }
  | { id: number; type: 'closeSource'; payload: CloseSourcePayload };

type PoseTraceWorkerResponse =
  | {
      id: number;
      ok: true;
      result: { sourceId: string; datasetName: string; demos: DemoInfo[] } | DemoRow[] | null;
    }
  | { id: number; ok: false; error: string };

interface OpenSourceEntry {
  datasetName: string;
  demos: DemoInfo[];
  h5File: H5WasmFile;
  cleanup: () => void;
}

const PLUGIN_ROOT = '/plugins';
const FILTER_PLUGIN_NAMES: Record<number, Plugin> = {
  307: Plugin.BZIP2,
  32000: Plugin.LZF,
  32001: Plugin.Blosc,
  32004: Plugin.LZ4,
  32008: Plugin.Bitshuffle,
  32013: Plugin.ZFP,
  32015: Plugin.Zstandard,
  32019: Plugin.JPEG,
  32026: Plugin.Blosc2,
};

const openSources = new Map<string, OpenSourceEntry>();
let modulePromise: Promise<H5Module> | null = null;

function sanitizeFilename(filename: string): string {
  return filename.replaceAll(/[^a-zA-Z0-9._-]/g, '_');
}

function stripExtension(filename: string): string {
  return filename.replace(/\.(hdf5|h5)$/i, '');
}

function uniqueName(prefix: string): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

async function ensureModule(): Promise<H5Module> {
  if (!modulePromise) {
    modulePromise = (async () => {
      const module = await h5wasm.ready;
      module.activate_throwing_error_handler();
      try {
        module.remove_plugin_search_path(0);
      } catch {
        // Ignore if the default search path is absent.
      }
      module.FS.mkdirTree(PLUGIN_ROOT);
      module.insert_plugin_search_path(PLUGIN_ROOT, 0);

      const pluginNames = [...new Set(Object.values(FILTER_PLUGIN_NAMES))];
      await Promise.all(
        pluginNames.map(async (pluginName) => {
          const pluginData = await getPlugin(pluginName);
          if (!pluginData) {
            return;
          }

          module.FS.writeFile(
            `${PLUGIN_ROOT}/libH5Z${pluginName}.so`,
            new Uint8Array(pluginData),
          );
        }),
      );

      return module;
    })();
  }

  return modulePromise;
}

function demoSortKey(name: string): [number, number, string] {
  if (name.startsWith('demo_')) {
    const suffix = name.slice('demo_'.length);
    if (/^\d+$/.test(suffix)) {
      return [0, Number(suffix), name];
    }
  }

  return [1, 0, name];
}

function optionalInt(value: unknown): number | null {
  if (value == null || value === '') {
    return null;
  }

  if (typeof value === 'number' && Number.isFinite(value)) {
    return Math.trunc(value);
  }

  if (typeof value === 'string') {
    const parsed = Number.parseInt(value, 10);
    return Number.isNaN(parsed) ? null : parsed;
  }

  return null;
}

function optionalBool(value: unknown): boolean | null {
  if (value == null || value === '') {
    return null;
  }

  if (typeof value === 'boolean') {
    return value;
  }

  if (typeof value === 'number') {
    return value !== 0;
  }

  if (typeof value === 'string') {
    const normalized = value.trim().toLowerCase();
    if (normalized === '0' || normalized === 'false') return false;
    if (normalized === '1' || normalized === 'true') return true;
    return normalized.length > 0;
  }

  return Boolean(value);
}

function roundFloat(value: number): number {
  return Math.round(value * 1_000_000) / 1_000_000;
}

function isGroup(entity: Entity | null): entity is H5WasmGroup {
  return entity instanceof h5wasm.Group;
}

function isDataset(entity: Entity | null): entity is H5WasmDataset {
  return entity instanceof h5wasm.Dataset;
}

function getAttributeValue(group: H5WasmGroup, name: string): unknown {
  return group.attrs[name]?.json_value;
}

function maybeChildGroup(group: H5WasmGroup, name: string): H5WasmGroup | null {
  const child = group.get(name);
  return isGroup(child) ? child : null;
}

function findDescendantGroup(group: H5WasmGroup, targetName: string): H5WasmGroup | null {
  const directChild = maybeChildGroup(group, targetName);
  if (directChild) {
    return directChild;
  }

  for (const childName of group.keys()) {
    const child = group.get(childName);
    if (!isGroup(child)) {
      continue;
    }

    if (child.path.split('/').pop() === targetName) {
      return child;
    }

    const nestedMatch = findDescendantGroup(child, targetName);
    if (nestedMatch) {
      return nestedMatch;
    }
  }

  return null;
}

function hasDemoChildren(group: H5WasmGroup): boolean {
  return group.keys().some((name) => name.startsWith('demo_'));
}

function findGroupWithDemoChildren(group: H5WasmGroup): H5WasmGroup | null {
  if (hasDemoChildren(group)) {
    return group;
  }

  for (const childName of group.keys()) {
    const child = group.get(childName);
    if (!isGroup(child)) {
      continue;
    }

    const nestedMatch = findGroupWithDemoChildren(child);
    if (nestedMatch) {
      return nestedMatch;
    }
  }

  return null;
}

function getDataGroup(h5File: H5WasmFile): H5WasmGroup {
  const rootDataGroup = h5File.get('data');
  if (isGroup(rootDataGroup) && hasDemoChildren(rootDataGroup)) {
    return rootDataGroup;
  }

  const nestedDataGroup = findDescendantGroup(h5File, 'data');
  if (nestedDataGroup && hasDemoChildren(nestedDataGroup)) {
    return nestedDataGroup;
  }

  const fallbackGroup = findGroupWithDemoChildren(h5File);
  if (fallbackGroup) {
    return fallbackGroup;
  }

  throw new Error(
    'Unsupported HDF5 schema: could not find a group containing demo_* entries.',
  );
}

function findPoseGroup(demoGroup: H5WasmGroup, name: string): H5WasmGroup | null {
  const searchRoots: H5WasmGroup[] = [];
  const obsGroup = maybeChildGroup(demoGroup, 'obs');
  if (obsGroup) {
    searchRoots.push(obsGroup);
  }
  searchRoots.push(demoGroup);

  const seen = new Set<string>();
  for (const root of searchRoots) {
    if (seen.has(root.path)) {
      continue;
    }
    seen.add(root.path);

    const match = findDescendantGroup(root, name);
    if (match) {
      return match;
    }
  }

  return null;
}

function isPoseShape(shape: number[] | null): shape is [number, number, number] {
  return Array.isArray(shape) && shape.length === 3 && shape[1] === 4 && shape[2] === 4;
}

function loadPoseArrays(
  group: H5WasmGroup | null,
  names: readonly string[],
): Record<string, PoseSeries> {
  const poseArrays: Record<string, PoseSeries> = {};

  if (!group) {
    return poseArrays;
  }

  for (const name of names) {
    const dataset = group.get(name);
    if (!isDataset(dataset) || !isPoseShape(dataset.shape)) {
      continue;
    }

    const values = dataset.to_array();
    if (!Array.isArray(values)) {
      continue;
    }

    poseArrays[name] = values as PoseSeries;
  }

  return poseArrays;
}

function extractXYZAtStep(
  poseSeries: PoseSeries | undefined,
  stepIdx: number,
): [number, number, number] | null {
  if (!poseSeries || stepIdx >= poseSeries.length) {
    return null;
  }

  const frame = poseSeries[stepIdx];
  const x = frame?.[0]?.[3];
  const y = frame?.[1]?.[3];
  const z = frame?.[2]?.[3];

  if (typeof x !== 'number' || typeof y !== 'number' || typeof z !== 'number') {
    return null;
  }

  return [roundFloat(x), roundFloat(y), roundFloat(z)];
}

function distanceXYZ(
  a: [number, number, number] | null,
  b: [number, number, number] | null,
): number | null {
  if (!a || !b) {
    return null;
  }

  return roundFloat(Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]));
}

function listDemos(h5File: H5WasmFile): DemoInfo[] {
  const dataGroup = getDataGroup(h5File);
  const demoNames = dataGroup.keys()
    .filter((name) => name.startsWith('demo_'))
    .sort((left, right) => {
      const leftKey = demoSortKey(left);
      const rightKey = demoSortKey(right);
      return (
        leftKey[0] - rightKey[0] ||
        leftKey[1] - rightKey[1] ||
        leftKey[2].localeCompare(rightKey[2])
      );
    });

  if (demoNames.length === 0) {
    throw new Error(
      `Unsupported HDF5 schema: no demo_* groups found under ${dataGroup.path}.`,
    );
  }

  return demoNames.map((demoName) => {
    const demoGroup = dataGroup.get(demoName);
    if (!isGroup(demoGroup)) {
      throw new Error(`Unsupported HDF5 schema: ${demoName} is not a group.`);
    }

    return {
      name: demoName,
      episode_index: optionalInt(demoName.split('_', 2)[1]),
      num_samples: optionalInt(getAttributeValue(demoGroup, 'num_samples')),
      success: optionalBool(getAttributeValue(demoGroup, 'success')),
      source_episode_index: optionalInt(
        getAttributeValue(demoGroup, 'source_episode_index'),
      ),
    };
  });
}

function buildDemoRows(entry: OpenSourceEntry, demoName: string): DemoRow[] {
  const dataGroup = getDataGroup(entry.h5File);
  const demoGroup = dataGroup.get(demoName);
  if (!isGroup(demoGroup)) {
    throw new Error(`Demo '${demoName}' not found in ${entry.datasetName}.`);
  }

  const eefPoseGroup = findPoseGroup(demoGroup, 'eef_pose');
  const objectPoseGroup = findPoseGroup(demoGroup, 'object_pose');

  const eefPose = loadPoseArrays(eefPoseGroup, TRACE_EEF_NAMES);
  const objectPose = loadPoseArrays(objectPoseGroup, TRACE_KEYPOINT_NAMES);

  if (Object.keys(eefPose).length === 0 && Object.keys(objectPose).length === 0) {
    throw new Error(`Demo '${demoName}' does not contain usable pose datasets.`);
  }

  const poseArrays = [...Object.values(eefPose), ...Object.values(objectPose)];
  const validLengths = poseArrays
    .filter((series) => series.every((frame) => Array.isArray(frame) && frame.length === 4))
    .map((series) => series.length);

  if (validLengths.length === 0) {
    throw new Error(`Demo '${demoName}' does not contain valid [T,4,4] pose arrays.`);
  }

  const numSteps = Math.max(...validLengths);
  const storedNumSamples = optionalInt(getAttributeValue(demoGroup, 'num_samples'));
  const successValue = optionalBool(getAttributeValue(demoGroup, 'success'));
  const episodeIndex = optionalInt(demoName.split('_', 2)[1]);
  const sourceEpisodeIndex = optionalInt(
    getAttributeValue(demoGroup, 'source_episode_index'),
  );

  const rows: DemoRow[] = [];
  for (let stepIdx = 0; stepIdx < numSteps; stepIdx += 1) {
    const row: DemoRow = {
      dataset_name: entry.datasetName,
      demo_name: demoName,
      step: stepIdx,
      env_id: 0,
      episode_index: episodeIndex,
      episode_step: stepIdx,
      source_episode_index: sourceEpisodeIndex,
      num_samples: storedNumSamples ?? numSteps,
      success: successValue == null ? null : Number(successValue),
      completed_attempts: null,
      completed_successes: null,
    };

    const eefPositions: Record<string, [number, number, number] | null> = {};
    const keypointPositions: Record<string, [number, number, number] | null> = {};

    for (const eefName of TRACE_EEF_NAMES) {
      const xyz = extractXYZAtStep(eefPose[eefName], stepIdx);
      eefPositions[eefName] = xyz;
      row[`eef_${eefName}_x`] = xyz?.[0] ?? null;
      row[`eef_${eefName}_y`] = xyz?.[1] ?? null;
      row[`eef_${eefName}_z`] = xyz?.[2] ?? null;
    }

    for (const keypointName of TRACE_KEYPOINT_NAMES) {
      const xyz = extractXYZAtStep(objectPose[keypointName], stepIdx);
      keypointPositions[keypointName] = xyz;
      row[`keypoint_${keypointName}_x`] = xyz?.[0] ?? null;
      row[`keypoint_${keypointName}_y`] = xyz?.[1] ?? null;
      row[`keypoint_${keypointName}_z`] = xyz?.[2] ?? null;
    }

    for (const [eefName, keypointNames] of Object.entries(TRACE_EEF_KEYPOINT_GROUPS)) {
      const eefXYZ = eefPositions[eefName];
      for (const keypointName of keypointNames) {
        row[`dist_${eefName}_to_${keypointName}_m`] = distanceXYZ(
          eefXYZ,
          keypointPositions[keypointName] ?? null,
        );
      }
    }

    for (const [metricName, srcName, dstName, threshold] of TRACE_SUCCESS_DISTANCE_SPECS) {
      const distance = distanceXYZ(
        keypointPositions[srcName] ?? null,
        keypointPositions[dstName] ?? null,
      );

      row[`dist_${metricName}_m`] = distance;
      row[`threshold_${metricName}_m`] = roundFloat(threshold);
      row[`pass_${metricName}`] = distance == null ? null : Number(distance <= threshold);
    }

    rows.push(row);
  }

  return rows;
}

async function openLocalSource(file: File) {
  const module = await ensureModule();
  const { FS } = module;
  const mountPoint = `/${uniqueName(`workerfs-${sanitizeFilename(file.name)}`)}`;
  const workerFsPath = `${mountPoint}/${file.name}`;

  FS.mkdir(mountPoint);
  FS.mount(FS.filesystems.WORKERFS, { files: [file] }, mountPoint);

  const h5File = new h5wasm.File(workerFsPath, 'r');
  const demos = listDemos(h5File);
  const sourceId = uniqueName('source');

  openSources.set(sourceId, {
    datasetName: stripExtension(file.name),
    demos,
    h5File,
    cleanup: () => {
      try {
        h5File.close();
      } catch {
        // Ignore repeated cleanup and best-effort close failures.
      }

      try {
        FS.unmount(mountPoint);
      } catch {
        // Ignore if the workerfs mount has already been removed.
      }

      try {
        FS.rmdir(mountPoint);
      } catch {
        // Ignore if the mount point directory is already gone.
      }
    },
  });

  return {
    sourceId,
    datasetName: stripExtension(file.name),
    demos,
  };
}

async function openRemoteSource(buffer: ArrayBuffer, name: string) {
  const module = await ensureModule();
  const { FS } = module;
  const fsName = `/${uniqueName(sanitizeFilename(name))}`;

  FS.writeFile(fsName, new Uint8Array(buffer), { flags: 'w+' });
  const h5File = new h5wasm.File(fsName, 'r');
  const demos = listDemos(h5File);
  const sourceId = uniqueName('source');

  openSources.set(sourceId, {
    datasetName: stripExtension(name),
    demos,
    h5File,
    cleanup: () => {
      try {
        h5File.close();
      } catch {
        // Ignore repeated cleanup and best-effort close failures.
      }

      try {
        FS.unlink(fsName);
      } catch {
        // Ignore if the temp file has already been removed.
      }
    },
  });

  return {
    sourceId,
    datasetName: stripExtension(name),
    demos,
  };
}

async function closeSource(sourceId: string) {
  const entry = openSources.get(sourceId);
  if (!entry) {
    return null;
  }

  entry.cleanup();
  openSources.delete(sourceId);
  return null;
}

async function handleRequest(message: PoseTraceWorkerRequest) {
  switch (message.type) {
    case 'openLocalSource':
      return openLocalSource(message.payload.file);
    case 'openRemoteSource':
      return openRemoteSource(message.payload.buffer, message.payload.name);
    case 'loadDemoRows': {
      const entry = openSources.get(message.payload.sourceId);
      if (!entry) {
        throw new Error('Pose Trace source is no longer available.');
      }
      return buildDemoRows(entry, message.payload.demoName);
    }
    case 'closeSource':
      return closeSource(message.payload.sourceId);
    default:
      throw new Error('Unsupported Pose Trace worker request.');
  }
}

self.onmessage = (event: MessageEvent<PoseTraceWorkerRequest>) => {
  const message = event.data;

  void handleRequest(message)
    .then((result) => {
      const response: PoseTraceWorkerResponse = {
        id: message.id,
        ok: true,
        result,
      };
      self.postMessage(response);
    })
    .catch((error: unknown) => {
      const response: PoseTraceWorkerResponse = {
        id: message.id,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      };
      self.postMessage(response);
    });
};
