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
import { DEMO_VIDEO_KEYS } from './types';
import type {
  DemoInfo,
  DemoRow,
  DatasetProcessingProgress,
  DatasetProcessingRequest,
  DatasetProcessingSourceInfo,
  DatasetProcessingResultMeta,
  DemoVideoInfo,
  DemoVideoKey,
} from './types';

type PoseSeries = number[][][];
type H5Module = Awaited<typeof h5wasm.ready>;
type H5WasmCreateDatasetArgs = Parameters<H5WasmGroup['create_dataset']>[0];
type H5WasmCompressionConfig = Pick<H5WasmCreateDatasetArgs, 'compression' | 'compression_opts'>;

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

type GetDatasetProcessingInfoPayload = {
  sourceId: string;
};

type ProcessDatasetPayload = DatasetProcessingRequest;

type ListDemoVideosPayload = {
  sourceId: string;
  demoName: string;
};

type LoadDemoVideoPayload = {
  sourceId: string;
  demoName: string;
  videoKey: DemoVideoKey;
};

type CloseSourcePayload = {
  sourceId: string;
};

type PoseTraceWorkerRequest =
  | { id: number; type: 'openLocalSource'; payload: OpenLocalSourcePayload }
  | { id: number; type: 'openRemoteSource'; payload: OpenRemoteSourcePayload }
  | { id: number; type: 'loadDemoRows'; payload: LoadDemoRowsPayload }
  | { id: number; type: 'getDatasetProcessingInfo'; payload: GetDatasetProcessingInfoPayload }
  | { id: number; type: 'processDataset'; payload: ProcessDatasetPayload }
  | { id: number; type: 'listDemoVideos'; payload: ListDemoVideosPayload }
  | { id: number; type: 'loadDemoVideo'; payload: LoadDemoVideoPayload }
  | { id: number; type: 'closeSource'; payload: CloseSourcePayload };

type LoadDemoVideoResult = DemoVideoInfo & {
  framesBuffer: ArrayBuffer;
};

type ProcessDatasetResult = DatasetProcessingResultMeta;

type PoseTraceWorkerResponse =
  | {
      id: number;
      ok: true;
      result:
        | { sourceId: string; datasetName: string; demos: DemoInfo[] }
        | DemoRow[]
        | DatasetProcessingSourceInfo
        | ProcessDatasetResult
        | DemoVideoInfo[]
        | LoadDemoVideoResult
        | null;
    }
  | { id: number; ok: false; error: string }
  | { id: number; type: 'progress'; progress: DatasetProcessingProgress }
  | { id: number; type: 'chunk'; data: ArrayBuffer; index: number; total: number };

interface WorkerSuccessResult {
  result:
    | { sourceId: string; datasetName: string; demos: DemoInfo[] }
    | DemoRow[]
    | DatasetProcessingSourceInfo
    | ProcessDatasetResult
    | DemoVideoInfo[]
    | LoadDemoVideoResult
    | null;
  transfer?: Transferable[];
}

interface OpenSourceEntry {
  datasetName: string;
  demos: DemoInfo[];
  h5File: H5WasmFile;
  cleanup: () => void;
}

interface DatasetCopyPlanItem {
  datasetName: string;
}

interface DatasetCopyPlanGroup {
  parentGroupSegments: string[];
  datasets: DatasetCopyPlanItem[];
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
const workerScope = self as unknown as {
  postMessage: (message: PoseTraceWorkerResponse, transfer?: Transferable[]) => void;
};

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

function maybeChildDataset(group: H5WasmGroup, name: string): H5WasmDataset | null {
  const child = group.get(name);
  return isDataset(child) ? child : null;
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

function findDescendantDataset(group: H5WasmGroup, targetName: string): H5WasmDataset | null {
  const directChild = maybeChildDataset(group, targetName);
  if (directChild) {
    return directChild;
  }

  for (const childName of group.keys()) {
    const child = group.get(childName);
    if (isDataset(child) && child.path.split('/').pop() === targetName) {
      return child;
    }

    if (!isGroup(child)) {
      continue;
    }

    const nestedMatch = findDescendantDataset(child, targetName);
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

function findVideoDataset(demoGroup: H5WasmGroup, name: DemoVideoKey): H5WasmDataset | null {
  const obsGroup = maybeChildGroup(demoGroup, 'obs');
  if (!obsGroup) {
    return null;
  }

  const directDataset = maybeChildDataset(obsGroup, name);
  if (directDataset) {
    return directDataset;
  }

  return findDescendantDataset(obsGroup, name);
}

function isPoseShape(shape: number[] | null): shape is [number, number, number] {
  return Array.isArray(shape) && shape.length === 3 && shape[1] === 4 && shape[2] === 4;
}

function isVideoShape(shape: number[] | null): shape is [number, number, number, number] {
  return (
    Array.isArray(shape)
    && shape.length === 4
    && Number.isInteger(shape[0])
    && Number.isInteger(shape[1])
    && Number.isInteger(shape[2])
    && [1, 3, 4].includes(shape[3] ?? 0)
  );
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

function getDemoGroup(entry: OpenSourceEntry, demoName: string): H5WasmGroup {
  const dataGroup = getDataGroup(entry.h5File);
  const demoGroup = dataGroup.get(demoName);
  if (!isGroup(demoGroup)) {
    throw new Error(`Demo '${demoName}' not found in ${entry.datasetName}.`);
  }

  return demoGroup;
}

function buildDemoRows(entry: OpenSourceEntry, demoName: string): DemoRow[] {
  const demoGroup = getDemoGroup(entry, demoName);

  const eefPoseGroup = findPoseGroup(demoGroup, 'eef_pose');
  const ikInputEefPoseGroup = findPoseGroup(demoGroup, 'ik_input_eef_pose');
  const eefPosePostStepGroup = findPoseGroup(demoGroup, 'eef_pose_post_step');
  const objectPoseGroup = findPoseGroup(demoGroup, 'object_pose');

  const eefPose = loadPoseArrays(eefPoseGroup, TRACE_EEF_NAMES);
  const ikInputEefPose = loadPoseArrays(ikInputEefPoseGroup, TRACE_EEF_NAMES);
  const eefPosePostStep = loadPoseArrays(eefPosePostStepGroup, TRACE_EEF_NAMES);
  const objectPose = loadPoseArrays(objectPoseGroup, TRACE_KEYPOINT_NAMES);

  if (
    Object.keys(eefPose).length === 0
    && Object.keys(ikInputEefPose).length === 0
    && Object.keys(eefPosePostStep).length === 0
    && Object.keys(objectPose).length === 0
  ) {
    throw new Error(`Demo '${demoName}' does not contain usable pose datasets.`);
  }

  const poseArrays = [
    ...Object.values(eefPose),
    ...Object.values(ikInputEefPose),
    ...Object.values(eefPosePostStep),
    ...Object.values(objectPose),
  ];
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

      const ikInputXYZ = extractXYZAtStep(ikInputEefPose[eefName], stepIdx);
      row[`ik_input_eef_${eefName}_x`] = ikInputXYZ?.[0] ?? null;
      row[`ik_input_eef_${eefName}_y`] = ikInputXYZ?.[1] ?? null;
      row[`ik_input_eef_${eefName}_z`] = ikInputXYZ?.[2] ?? null;

      const postStepXYZ = extractXYZAtStep(eefPosePostStep[eefName], stepIdx);
      row[`eef_post_step_${eefName}_x`] = postStepXYZ?.[0] ?? null;
      row[`eef_post_step_${eefName}_y`] = postStepXYZ?.[1] ?? null;
      row[`eef_post_step_${eefName}_z`] = postStepXYZ?.[2] ?? null;
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

interface H5AttributeOwner {
  path: string;
  attrs: Record<string, {
    value: unknown;
    json_value: unknown;
    shape: number[] | null;
    dtype: Parameters<H5WasmGroup['create_attribute']>[3];
  }>;
  create_attribute: (
    name: string,
    data: Parameters<H5WasmGroup['create_attribute']>[1],
    shape?: number[] | null,
    dtype?: Parameters<H5WasmGroup['create_attribute']>[3],
  ) => void;
}

function collectRelativeDatasetPaths(
  group: H5WasmGroup,
  prefix: string,
  output: Set<string>,
) {
  for (const childName of group.keys()) {
    const child = group.get(childName);
    const childPath = prefix ? `${prefix}/${childName}` : childName;

    if (isDataset(child)) {
      output.add(childPath);
      continue;
    }

    if (isGroup(child)) {
      collectRelativeDatasetPaths(child, childPath, output);
    }
  }
}

function listDatasetProcessingInfo(entry: OpenSourceEntry): DatasetProcessingSourceInfo {
  const keyCounts = new Map<string, number>();

  for (const demo of entry.demos) {
    const demoGroup = getDemoGroup(entry, demo.name);
    const demoKeys = new Set<string>();
    collectRelativeDatasetPaths(demoGroup, '', demoKeys);

    for (const path of demoKeys) {
      keyCounts.set(path, (keyCounts.get(path) ?? 0) + 1);
    }
  }

  return {
    keyPaths: [...keyCounts.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([path, availableInDemoCount]) => ({
        path,
        availableInDemoCount,
      })),
  };
}

function copyAttributes(
  source: H5AttributeOwner,
  target: H5AttributeOwner,
  excludedNames: ReadonlySet<string> = new Set(),
) {
  for (const [name, attribute] of Object.entries(source.attrs)) {
    if (excludedNames.has(name)) {
      continue;
    }

    const value = attribute.value ?? attribute.json_value;
    if (value == null) {
      continue;
    }

    try {
      target.create_attribute(
        name,
        value as Parameters<H5WasmGroup['create_attribute']>[1],
        attribute.shape,
        attribute.dtype,
      );
    } catch {
      // Ignore unsupported attribute variants and preserve the rest of the structure.
    }
  }
}

function getOrderedSourceEntries(sourceIds: string[]): OpenSourceEntry[] {
  if (sourceIds.length === 0) {
    throw new Error('Select at least one source dataset.');
  }

  return sourceIds.map((sourceId) => {
    const entry = openSources.get(sourceId);
    if (!entry) {
      throw new Error('One of the selected source datasets is no longer available.');
    }

    return entry;
  });
}

function getCutDemoNames(
  entry: OpenSourceEntry,
  cutRange: DatasetProcessingRequest['cutRange'],
): string[] {
  const demoNames = entry.demos.map((demo) => demo.name);
  if (!cutRange) {
    return demoNames;
  }

  const startIndex = demoNames.indexOf(cutRange.startDemoName);
  const endIndex = demoNames.indexOf(cutRange.endDemoName);

  if (startIndex < 0 || endIndex < 0) {
    throw new Error('The selected demo range is no longer valid.');
  }

  if (startIndex > endIndex) {
    throw new Error('The cut start demo must come before the end demo.');
  }

  return demoNames.slice(startIndex, endIndex + 1);
}

function ensureChildGroup(parent: H5WasmGroup, name: string): H5WasmGroup {
  const existing = parent.get(name);
  if (isGroup(existing)) {
    return existing;
  }

  return parent.create_group(name, true);
}

function ensureTargetGroupPath(
  targetDemoGroup: H5WasmGroup,
  sourceDemoGroup: H5WasmGroup,
  relativeGroupSegments: readonly string[],
  copiedGroupPaths: Set<string>,
): {
  sourceGroup: H5WasmGroup;
  targetGroup: H5WasmGroup;
} | null {
  let targetGroup = targetDemoGroup;
  let sourceGroup = sourceDemoGroup;

  if (relativeGroupSegments.length === 0) {
    return { sourceGroup, targetGroup };
  }

  for (const segment of relativeGroupSegments) {
    const nextSource = sourceGroup.get(segment);
    if (!isGroup(nextSource)) {
      // Group does not exist in this demo — caller should skip this dataset.
      return null;
    }

    targetGroup = ensureChildGroup(targetGroup, segment);
    sourceGroup = nextSource;

    const currentPath = targetGroup.path;
    if (!copiedGroupPaths.has(currentPath)) {
      copyAttributes(sourceGroup, targetGroup);
      copiedGroupPaths.add(currentPath);
    }
  }

  return { sourceGroup, targetGroup };
}

function getDatasetCopyValue(
  dataset: H5WasmDataset,
): H5WasmCreateDatasetArgs['data'] {
  const value = dataset.value;
  if (value != null) {
    return value as H5WasmCreateDatasetArgs['data'];
  }

  const jsonValue = dataset.json_value;
  if (jsonValue != null) {
    return jsonValue as H5WasmCreateDatasetArgs['data'];
  }

  throw new Error(`Dataset '${dataset.path}' does not expose copyable data.`);
}

function buildDatasetCopyPlan(selectedKeys: readonly string[]): DatasetCopyPlanGroup[] {
  const planByParentGroup = new Map<string, DatasetCopyPlanGroup>();

  for (const keyPath of selectedKeys) {
    const splitIndex = keyPath.lastIndexOf('/');
    const parentGroupPath = splitIndex >= 0 ? keyPath.slice(0, splitIndex) : '';
    const datasetName = splitIndex >= 0 ? keyPath.slice(splitIndex + 1) : keyPath;

    let groupPlan = planByParentGroup.get(parentGroupPath);
    if (!groupPlan) {
      groupPlan = {
        parentGroupSegments: parentGroupPath ? parentGroupPath.split('/') : [],
        datasets: [],
      };
      planByParentGroup.set(parentGroupPath, groupPlan);
    }

    groupPlan.datasets.push({ datasetName });
  }

  return [...planByParentGroup.values()];
}

function getDatasetCompressionConfig(dataset: H5WasmDataset): H5WasmCompressionConfig {
  for (let index = dataset.filters.length - 1; index >= 0; index -= 1) {
    const filter = dataset.filters[index];

    if (filter.id === 1) {
      const compressionLevel = filter.cd_values[0];
      return Number.isFinite(compressionLevel)
        ? {
            compression: 'gzip',
            compression_opts: compressionLevel,
          }
        : {
            compression: 'gzip',
          };
    }

    if (FILTER_PLUGIN_NAMES[filter.id]) {
      return filter.cd_values.length > 0
        ? {
            compression: filter.id,
            compression_opts: [...filter.cd_values],
          }
        : {
            compression: filter.id,
          };
    }
  }

  return {};
}

// Datasets larger than this threshold (in bytes) are copied in chunks along
// the first axis to avoid allocating the entire dataset in WASM memory at once.
const CHUNKED_COPY_BYTE_THRESHOLD = 50 * 1024 * 1024; // 50 MB
const CHUNKED_COPY_ROW_BATCH = 50;

function estimateDatasetBytes(dataset: H5WasmDataset): number {
  const { shape, size } = dataset.metadata;
  if (!shape || shape.length === 0) {
    return 0;
  }

  const totalElements = shape.reduce((acc, dim) => acc * dim, 1);
  return totalElements * size;
}

function copyDatasetChunked(
  source: H5WasmDataset,
  targetGroup: H5WasmGroup,
  name: string,
  onSliceProgress?: (copiedRows: number, totalRows: number) => void,
) {
  const metadata = source.metadata;
  const { shape } = metadata;
  if (!shape || shape.length === 0 || shape[0] === 0) {
    return;
  }

  const totalRows = shape[0];
  const restShape = shape.slice(1);
  const compressionConfig = getDatasetCompressionConfig(source);

  // Read the first batch to create the dataset.
  const firstBatchSize = Math.min(CHUNKED_COPY_ROW_BATCH, totalRows);
  const firstSlice = source.slice([[0, firstBatchSize]]);
  if (firstSlice == null) {
    return;
  }

  const targetDataset = targetGroup.create_dataset({
    name,
    data: firstSlice as H5WasmCreateDatasetArgs['data'],
    shape: [firstBatchSize, ...restShape],
    dtype: source.dtype,
    maxshape: [null, ...restShape],
    chunks: metadata.chunks,
    track_order: true,
    ...compressionConfig,
  });

  onSliceProgress?.(firstBatchSize, totalRows);

  // Write remaining batches by resizing and writing slices.
  for (let offset = firstBatchSize; offset < totalRows; offset += CHUNKED_COPY_ROW_BATCH) {
    const end = Math.min(offset + CHUNKED_COPY_ROW_BATCH, totalRows);
    const batchData = source.slice([[offset, end]]);
    if (batchData == null) {
      break;
    }

    targetDataset.resize([end, ...restShape]);
    targetDataset.write_slice([[offset, end]], batchData);
    onSliceProgress?.(end, totalRows);
  }

  // Resize to the exact final shape (in case maxshape was unlimited).
  targetDataset.resize(shape);
  copyAttributes(source, targetDataset);
}

type SliceProgressCallback = (datasetPath: string, copiedRows: number, totalRows: number) => void;

function copySelectedDatasetsForDemo(
  sourceDemoGroup: H5WasmGroup,
  targetDemoGroup: H5WasmGroup,
  copyPlan: readonly DatasetCopyPlanGroup[],
  onSliceProgress?: SliceProgressCallback,
) {
  const copiedGroupPaths = new Set<string>([targetDemoGroup.path]);

  for (const groupPlan of copyPlan) {
    const resolved = ensureTargetGroupPath(
      targetDemoGroup,
      sourceDemoGroup,
      groupPlan.parentGroupSegments,
      copiedGroupPaths,
    );

    if (!resolved) {
      // Parent group does not exist in this demo — skip all datasets under it.
      continue;
    }

    const { sourceGroup, targetGroup } = resolved;

    for (const datasetPlan of groupPlan.datasets) {
      const sourceEntity = sourceGroup.get(datasetPlan.datasetName);
      if (!isDataset(sourceEntity)) {
        continue;
      }

      const estimatedBytes = estimateDatasetBytes(sourceEntity);
      const metadata = sourceEntity.metadata;

      if (estimatedBytes > CHUNKED_COPY_BYTE_THRESHOLD && metadata.shape && metadata.shape.length > 0 && metadata.chunks) {
        // Large dataset: copy in chunks along the first axis.
        copyDatasetChunked(
          sourceEntity,
          targetGroup,
          datasetPlan.datasetName,
          onSliceProgress
            ? (copiedRows, totalRows) => {
                onSliceProgress(datasetPlan.datasetName, copiedRows, totalRows);
              }
            : undefined,
        );
      } else {
        // Small dataset: copy in one shot.
        const targetDataset = targetGroup.create_dataset({
          name: datasetPlan.datasetName,
          data: getDatasetCopyValue(sourceEntity),
          shape: metadata.shape,
          dtype: sourceEntity.dtype,
          maxshape: metadata.maxshape,
          chunks: metadata.chunks,
          track_order: true,
          ...getDatasetCompressionConfig(sourceEntity),
        });

        copyAttributes(sourceEntity, targetDataset);
      }
    }
  }
}

function normalizeOutputFileName(fileName: string): string {
  const trimmed = fileName.trim();
  if (trimmed.length === 0) {
    return 'processed-dataset.hdf5';
  }

  return /\.(hdf5|h5)$/i.test(trimmed) ? trimmed : `${trimmed}.hdf5`;
}

const OUTPUT_CHUNK_SIZE = 64 * 1024 * 1024; // 64 MB

async function processDataset(
  request: DatasetProcessingRequest,
  messageId: number,
): Promise<ProcessDatasetResult> {
  const module = await ensureModule();
  const { FS } = module;
  const entries = getOrderedSourceEntries(request.orderedSourceIds);
  const selectedKeys = [...new Set(request.selectedKeys)].sort((left, right) => left.localeCompare(right));
  const copyPlan = buildDatasetCopyPlan(selectedKeys);

  if (selectedKeys.length === 0) {
    throw new Error('Select at least one key to include in the processed dataset.');
  }

  // Pre-count total demos for accurate progress reporting.
  let overallDemoCount = 0;
  const demoNamesByEntry: string[][] = [];
  for (const entry of entries) {
    const names = request.operation === 'cut'
      ? getCutDemoNames(entry, request.cutRange)
      : entry.demos.map((demo) => demo.name);
    demoNamesByEntry.push(names);
    overallDemoCount += names.length;
  }

  function reportProgress(progress: DatasetProcessingProgress) {
    workerScope.postMessage({ id: messageId, type: 'progress', progress });
  }

  const outputName = normalizeOutputFileName(request.fileName);
  const outputPath = `/${uniqueName(sanitizeFilename(outputName))}`;
  const outputFile = new h5wasm.File(outputPath, 'w', { track_order: true });

  try {
    copyAttributes(entries[0].h5File, outputFile);

    const firstDataGroup = getDataGroup(entries[0].h5File);
    const outputDataGroup = outputFile.create_group('data', true);
    copyAttributes(firstDataGroup, outputDataGroup, new Set(['total']));

    let outputDemoIndex = 0;
    let totalSamples = 0;
    let overallDemoIndex = 0;

    for (const [entryIndex, entry] of entries.entries()) {
      const sourceDemoNames = demoNamesByEntry[entryIndex];

      for (const sourceDemoName of sourceDemoNames) {
        reportProgress({
          phase: 'copying',
          overallDemoIndex,
          overallDemoCount,
          currentSourceName: entry.datasetName,
          currentDemoName: sourceDemoName,
        });

        const sourceDemoGroup = getDemoGroup(entry, sourceDemoName);
        const targetDemoGroup = outputDataGroup.create_group(`demo_${outputDemoIndex}`, true);

        copyAttributes(sourceDemoGroup, targetDemoGroup);
        const capturedDemoIndex = overallDemoIndex;
        copySelectedDatasetsForDemo(sourceDemoGroup, targetDemoGroup, copyPlan, (datasetPath, copiedRows, totalRows) => {
          reportProgress({
            phase: 'copying',
            overallDemoIndex: capturedDemoIndex,
            overallDemoCount,
            currentSourceName: entry.datasetName,
            currentDemoName: sourceDemoName,
            datasetDetail: { path: datasetPath, copiedRows, totalRows },
          });
        });

        totalSamples += optionalInt(getAttributeValue(sourceDemoGroup, 'num_samples')) ?? 0;
        outputDemoIndex += 1;
        overallDemoIndex += 1;
      }

      // Flush after each source file's demos to reduce memory pressure.
      outputFile.flush();
    }

    if (outputDemoIndex === 0) {
      throw new Error('No demos matched the selected processing settings.');
    }

    if ('total' in firstDataGroup.attrs) {
      outputDataGroup.create_attribute('total', totalSamples);
    }

    reportProgress({
      phase: 'flushing',
      overallDemoIndex: overallDemoCount,
      overallDemoCount,
      currentSourceName: '',
      currentDemoName: '',
    });

    outputFile.flush();
    outputFile.close();

    // Stream the output file in chunks instead of loading it all at once.
    reportProgress({
      phase: 'streaming',
      overallDemoIndex: overallDemoCount,
      overallDemoCount,
      currentSourceName: '',
      currentDemoName: '',
    });

    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access -- Emscripten FS.stat is untyped
    const totalSize: number = FS.stat(outputPath).size;
    const totalChunks = Math.ceil(totalSize / OUTPUT_CHUNK_SIZE);
    const stream = FS.open(outputPath, 'r');

    try {
      for (let i = 0; i < totalChunks; i++) {
        const position = i * OUTPUT_CHUNK_SIZE;
        const size = Math.min(OUTPUT_CHUNK_SIZE, totalSize - position);
        const buffer = new Uint8Array(size);
        FS.read(stream, buffer, 0, size, position);

        const arrayBuffer: ArrayBuffer = buffer.buffer.slice(
          buffer.byteOffset,
          buffer.byteOffset + buffer.byteLength,
        );

        workerScope.postMessage(
          { id: messageId, type: 'chunk', data: arrayBuffer, index: i, total: totalChunks },
          [arrayBuffer],
        );
      }
    } finally {
      FS.close(stream);
    }

    return {
      fileName: outputName,
      demoCount: outputDemoIndex,
      selectedKeyCount: selectedKeys.length,
    };
  } finally {
    try {
      outputFile.close();
    } catch {
      // Ignore repeated close attempts.
    }

    try {
      FS.unlink(outputPath);
    } catch {
      // Ignore if the temp output file is already removed.
    }
  }
}

function listDemoVideoInfo(entry: OpenSourceEntry, demoName: string): DemoVideoInfo[] {
  const demoGroup = getDemoGroup(entry, demoName);
  const videos: DemoVideoInfo[] = [];

  for (const key of DEMO_VIDEO_KEYS) {
    const dataset = findVideoDataset(demoGroup, key);
    if (!dataset || !isVideoShape(dataset.shape)) {
      continue;
    }

    const [frameCount, height, width, channels] = dataset.shape;
    videos.push({
      key,
      path: dataset.path,
      frameCount,
      height,
      width,
      channels,
    });
  }

  return videos;
}

function datasetToUint8Array(dataset: H5WasmDataset): Uint8Array {
  const value = dataset.value;

  if (value instanceof Uint8Array) {
    return new Uint8Array(value);
  }

  if (ArrayBuffer.isView(value)) {
    const copied = new Uint8Array(value.byteLength);
    copied.set(new Uint8Array(value.buffer, value.byteOffset, value.byteLength));
    return copied;
  }

  throw new Error(`Dataset '${dataset.path}' is not backed by byte data.`);
}

function loadDemoVideo(
  entry: OpenSourceEntry,
  demoName: string,
  videoKey: DemoVideoKey,
): LoadDemoVideoResult {
  const demoGroup = getDemoGroup(entry, demoName);
  const dataset = findVideoDataset(demoGroup, videoKey);

  if (!dataset || !isVideoShape(dataset.shape)) {
    throw new Error(
      `Demo '${demoName}' does not contain a supported video dataset at obs/${videoKey}.`,
    );
  }

  const [frameCount, height, width, channels] = dataset.shape;
  const frames = datasetToUint8Array(dataset);
  const framesBuffer = frames.buffer.slice(
    frames.byteOffset,
    frames.byteOffset + frames.byteLength,
  ) as ArrayBuffer;

  return {
    key: videoKey,
    path: dataset.path,
    frameCount,
    height,
    width,
    channels,
    framesBuffer,
  };
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

async function handleRequest(message: PoseTraceWorkerRequest): Promise<WorkerSuccessResult> {
  switch (message.type) {
    case 'openLocalSource':
      return { result: await openLocalSource(message.payload.file) };
    case 'openRemoteSource':
      return { result: await openRemoteSource(message.payload.buffer, message.payload.name) };
    case 'loadDemoRows': {
      const entry = openSources.get(message.payload.sourceId);
      if (!entry) {
        throw new Error('Pose Trace source is no longer available.');
      }
      return { result: buildDemoRows(entry, message.payload.demoName) };
    }
    case 'getDatasetProcessingInfo': {
      const entry = openSources.get(message.payload.sourceId);
      if (!entry) {
        throw new Error('Pose Trace source is no longer available.');
      }
      return { result: listDatasetProcessingInfo(entry) };
    }
    case 'processDataset': {
      const result = await processDataset(message.payload, message.id);
      return { result };
    }
    case 'listDemoVideos': {
      const entry = openSources.get(message.payload.sourceId);
      if (!entry) {
        throw new Error('Pose Trace source is no longer available.');
      }
      return { result: listDemoVideoInfo(entry, message.payload.demoName) };
    }
    case 'loadDemoVideo': {
      const entry = openSources.get(message.payload.sourceId);
      if (!entry) {
        throw new Error('Pose Trace source is no longer available.');
      }

      const result = loadDemoVideo(
        entry,
        message.payload.demoName,
        message.payload.videoKey,
      );

      return {
        result,
        transfer: [result.framesBuffer],
      };
    }
    case 'closeSource':
      return { result: await closeSource(message.payload.sourceId) };
    default:
      throw new Error('Unsupported Pose Trace worker request.');
  }
}

self.onmessage = (event: MessageEvent<PoseTraceWorkerRequest>) => {
  const message = event.data;

  void handleRequest(message)
    .then(({ result, transfer = [] }) => {
      const response: PoseTraceWorkerResponse = {
        id: message.id,
        ok: true,
        result,
      };
      workerScope.postMessage(response, transfer);
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
