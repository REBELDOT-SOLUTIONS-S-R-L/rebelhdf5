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
  DatasetProcessingRequest,
  DatasetProcessingResult,
  DatasetProcessingSourceInfo,
  DemoVideoInfo,
  DemoVideoKey,
} from './types';

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

type ProcessDatasetResult = Omit<DatasetProcessingResult, 'fileBuffer'> & {
  fileBuffer: ArrayBuffer;
};

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
  | { id: number; ok: false; error: string };

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
  relativeGroupPath: string,
  copiedGroupPaths: Set<string>,
): H5WasmGroup {
  let targetGroup = targetDemoGroup;
  let sourceGroup = sourceDemoGroup;

  if (!relativeGroupPath) {
    return targetGroup;
  }

  for (const segment of relativeGroupPath.split('/')) {
    const nextSource = sourceGroup.get(segment);
    if (!isGroup(nextSource)) {
      throw new Error(
        `Expected '${sourceGroup.path}/${segment}' to be a group while copying dataset structure.`,
      );
    }

    targetGroup = ensureChildGroup(targetGroup, segment);
    sourceGroup = nextSource;

    const currentPath = targetGroup.path;
    if (!copiedGroupPaths.has(currentPath)) {
      copyAttributes(sourceGroup, targetGroup);
      copiedGroupPaths.add(currentPath);
    }
  }

  return targetGroup;
}

function getDatasetCopyValue(
  dataset: H5WasmDataset,
): Parameters<H5WasmGroup['create_dataset']>[0]['data'] {
  const value = dataset.value;
  if (value != null) {
    return value as Parameters<H5WasmGroup['create_dataset']>[0]['data'];
  }

  const jsonValue = dataset.json_value;
  if (jsonValue != null) {
    return jsonValue as Parameters<H5WasmGroup['create_dataset']>[0]['data'];
  }

  throw new Error(`Dataset '${dataset.path}' does not expose copyable data.`);
}

function copySelectedDatasetsForDemo(
  sourceDemoGroup: H5WasmGroup,
  targetDemoGroup: H5WasmGroup,
  selectedKeys: readonly string[],
) {
  const copiedGroupPaths = new Set<string>([targetDemoGroup.path]);

  for (const keyPath of selectedKeys) {
    const sourceEntity = sourceDemoGroup.get(keyPath);
    if (!isDataset(sourceEntity)) {
      continue;
    }

    const pathParts = keyPath.split('/');
    const datasetName = pathParts[pathParts.length - 1];
    const parentGroupPath = pathParts.slice(0, -1).join('/');
    const targetParentGroup = ensureTargetGroupPath(
      targetDemoGroup,
      sourceDemoGroup,
      parentGroupPath,
      copiedGroupPaths,
    );

    const metadata = sourceEntity.metadata;
    const targetDataset = targetParentGroup.create_dataset({
      name: datasetName,
      data: getDatasetCopyValue(sourceEntity),
      shape: metadata.shape,
      dtype: sourceEntity.dtype,
      maxshape: metadata.maxshape,
      chunks: metadata.chunks,
      track_order: true,
    });

    copyAttributes(sourceEntity, targetDataset);
  }
}

function normalizeOutputFileName(fileName: string): string {
  const trimmed = fileName.trim();
  if (trimmed.length === 0) {
    return 'processed-dataset.hdf5';
  }

  return /\.(hdf5|h5)$/i.test(trimmed) ? trimmed : `${trimmed}.hdf5`;
}

async function processDataset(
  request: DatasetProcessingRequest,
): Promise<ProcessDatasetResult> {
  const module = await ensureModule();
  const { FS } = module;
  const entries = getOrderedSourceEntries(request.orderedSourceIds);
  const selectedKeys = [...new Set(request.selectedKeys)].sort((left, right) => left.localeCompare(right));

  if (selectedKeys.length === 0) {
    throw new Error('Select at least one key to include in the processed dataset.');
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

    for (const entry of entries) {
      const sourceDemoNames = request.operation === 'cut'
        ? getCutDemoNames(entry, request.cutRange)
        : entry.demos.map((demo) => demo.name);

      for (const sourceDemoName of sourceDemoNames) {
        const sourceDemoGroup = getDemoGroup(entry, sourceDemoName);
        const targetDemoGroup = outputDataGroup.create_group(`demo_${outputDemoIndex}`, true);

        copyAttributes(sourceDemoGroup, targetDemoGroup);
        copySelectedDatasetsForDemo(sourceDemoGroup, targetDemoGroup, selectedKeys);

        totalSamples += optionalInt(getAttributeValue(sourceDemoGroup, 'num_samples')) ?? 0;
        outputDemoIndex += 1;
      }
    }

    if (outputDemoIndex === 0) {
      throw new Error('No demos matched the selected processing settings.');
    }

    if ('total' in firstDataGroup.attrs) {
      outputDataGroup.create_attribute('total', totalSamples);
    }

    outputFile.flush();
    outputFile.close();

    const fileBytes = FS.readFile(outputPath);
    const fileBuffer = fileBytes.buffer.slice(
      fileBytes.byteOffset,
      fileBytes.byteOffset + fileBytes.byteLength,
    ) as ArrayBuffer;

    return {
      fileName: outputName,
      demoCount: outputDemoIndex,
      selectedKeyCount: selectedKeys.length,
      fileBuffer,
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
      const result = await processDataset(message.payload);
      return {
        result,
        transfer: [result.fileBuffer],
      };
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
