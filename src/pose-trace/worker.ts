import h5wasm, {
  type Dataset as H5WasmDataset,
  type Entity,
  type File as H5WasmFile,
  type Group as H5WasmGroup,
} from 'h5wasm';
import { Plugin } from '@h5web/h5wasm';

import { getPlugin } from '../plugin-utils';
import { OBJECT_DISTRIBUTION_ANCHORS } from './types';
import type {
  ArticulationEndEffector,
  ArticulationJoint,
  ArticulationSegment,
  ObjectDistributionAnchor,
  ObjectDistributionSourceDiagnostics,
  ObjectDistributionCategory,
  ObjectDistributionPoint,
  ObjectDistributionRequest,
  ObjectDistributionResult,
  ObjectDistributionSourceDetail,
  DemoInfo,
  DemoRow,
  DatasetComparisonValue,
  DatasetComparisonValuesResult,
  DatasetProcessingProgress,
  DatasetProcessingRequest,
  DatasetProcessingSourceInfo,
  DatasetProcessingResultMeta,
  DemoVideoInfo,
  DemoVideoKey,
  ParsedArticulation,
  SourceFeatureCapabilities,
} from './types';

type PoseSeries = number[][][];
type H5Module = Awaited<typeof h5wasm.ready>;
type H5WasmCreateDatasetArgs = Parameters<H5WasmGroup['create_dataset']>[0];
type H5WasmCompressionConfig = Pick<
  H5WasmCreateDatasetArgs,
  'compression' | 'compression_opts'
>;

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

type InspectSourceFeaturesPayload = {
  sourceId: string;
};

type LoadDatasetComparisonValuesPayload = {
  sourceId: string;
  demoName: string;
  keyPaths: string[];
};

type LoadObjectDistributionPayload = ObjectDistributionRequest;
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
  | {
      id: number;
      type: 'getDatasetProcessingInfo';
      payload: GetDatasetProcessingInfoPayload;
    }
  | {
      id: number;
      type: 'inspectSourceFeatures';
      payload: InspectSourceFeaturesPayload;
    }
  | {
      id: number;
      type: 'loadDatasetComparisonValues';
      payload: LoadDatasetComparisonValuesPayload;
    }
  | {
      id: number;
      type: 'loadObjectDistribution';
      payload: LoadObjectDistributionPayload;
    }
  | { id: number; type: 'processDataset'; payload: ProcessDatasetPayload }
  | { id: number; type: 'listDemoVideos'; payload: ListDemoVideosPayload }
  | { id: number; type: 'loadDemoVideo'; payload: LoadDemoVideoPayload }
  | { id: number; type: 'closeSource'; payload: CloseSourcePayload };

type LoadDemoVideoResult = DemoVideoInfo & {
  framesBuffer: ArrayBuffer;
};

type ProcessDatasetResult = DatasetProcessingResultMeta;

type OpenSourceResultPayload = {
  sourceId: string;
  datasetName: string;
  demos: DemoInfo[];
  articulation: ParsedArticulation | null;
};

type PoseTraceWorkerResponse =
  | {
      id: number;
      ok: true;
      result:
        | OpenSourceResultPayload
        | DemoRow[]
        | DatasetProcessingSourceInfo
        | SourceFeatureCapabilities
        | DatasetComparisonValuesResult
        | ObjectDistributionResult
        | ProcessDatasetResult
        | DemoVideoInfo[]
        | LoadDemoVideoResult
        | null;
    }
  | { id: number; ok: false; error: string }
  | { id: number; type: 'progress'; progress: DatasetProcessingProgress }
  | {
      id: number;
      type: 'chunk';
      data: ArrayBuffer;
      index: number;
      total: number;
    };

interface WorkerSuccessResult {
  result:
    | OpenSourceResultPayload
    | DemoRow[]
    | DatasetProcessingSourceInfo
    | SourceFeatureCapabilities
    | DatasetComparisonValuesResult
    | ObjectDistributionResult
    | ProcessDatasetResult
    | DemoVideoInfo[]
    | LoadDemoVideoResult
    | null;
  transfer?: Transferable[];
}

interface OpenSourceEntry {
  datasetName: string;
  demos: DemoInfo[];
  articulation: ParsedArticulation | null;
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

interface TeleopSource {
  teleopId: string;
  datasetName: string;
  demoName: string;
  x: number;
  y: number;
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
  postMessage: (
    message: PoseTraceWorkerResponse,
    transfer?: Transferable[],
  ) => void;
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

function maybeChildDataset(
  group: H5WasmGroup,
  name: string,
): H5WasmDataset | null {
  const child = group.get(name);
  return isDataset(child) ? child : null;
}

function findDescendantGroup(
  group: H5WasmGroup,
  targetName: string,
): H5WasmGroup | null {
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

function findDescendantDataset(
  group: H5WasmGroup,
  targetName: string,
): H5WasmDataset | null {
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

function findPoseGroup(
  demoGroup: H5WasmGroup,
  name: string,
): H5WasmGroup | null {
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

function findVideoDataset(
  demoGroup: H5WasmGroup,
  name: DemoVideoKey,
): H5WasmDataset | null {
  const obsGroup = maybeChildGroup(demoGroup, 'obs');
  if (!obsGroup) {
    return null;
  }

  // New standard schema: cameras live under `obs/cameras/<name>`.
  const camerasGroup = maybeChildGroup(obsGroup, 'cameras');
  if (camerasGroup) {
    const cameraDataset = maybeChildDataset(camerasGroup, name);
    if (cameraDataset) {
      return cameraDataset;
    }
  }

  // Legacy fallback: camera datasets stored directly under `obs/`.
  const directDataset = maybeChildDataset(obsGroup, name);
  if (directDataset) {
    return directDataset;
  }

  return findDescendantDataset(obsGroup, name);
}

function listCameraNames(demoGroup: H5WasmGroup): string[] {
  const obsGroup = maybeChildGroup(demoGroup, 'obs');
  if (!obsGroup) {
    return [];
  }

  const names = new Set<string>();

  // New standard schema: `obs/cameras/<name>` (T, H, W, C) datasets.
  const camerasGroup = maybeChildGroup(obsGroup, 'cameras');
  if (camerasGroup) {
    for (const childName of camerasGroup.keys()) {
      const child = camerasGroup.get(childName);
      if (isDataset(child) && isVideoShape(child.shape)) {
        names.add(childName);
      }
    }
  }

  // Legacy schema: video datasets stored directly under `obs/`.
  for (const childName of obsGroup.keys()) {
    const child = obsGroup.get(childName);
    if (isDataset(child) && isVideoShape(child.shape)) {
      names.add(childName);
    }
  }

  return [...names];
}

function humanizeCameraLabel(name: string): string {
  return name
    .replaceAll(/[_-]+/g, ' ')
    .replaceAll(/\s+/g, ' ')
    .trim()
    .split(' ')
    .map((word) =>
      word.length === 0 ? word : word[0].toUpperCase() + word.slice(1),
    )
    .join(' ');
}

function isPoseShape(
  shape: number[] | null,
): shape is [number, number, number] {
  return (
    Array.isArray(shape) &&
    shape.length === 3 &&
    shape[1] === 4 &&
    shape[2] === 4
  );
}

function isVideoShape(
  shape: number[] | null,
): shape is [number, number, number, number] {
  return (
    Array.isArray(shape) &&
    shape.length === 4 &&
    Number.isInteger(shape[0]) &&
    Number.isInteger(shape[1]) &&
    Number.isInteger(shape[2]) &&
    [1, 3, 4].includes(shape[3] ?? 0)
  );
}

function loadPoseArrays(
  group: H5WasmGroup | null,
  names?: readonly string[],
): Record<string, PoseSeries> {
  const poseArrays: Record<string, PoseSeries> = {};

  if (!group) {
    return poseArrays;
  }

  const candidateNames = names ?? group.keys();
  for (const name of candidateNames) {
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

function loadNestedPoseArrays(
  parentGroup: H5WasmGroup | null,
): Record<string, PoseSeries> {
  const poseArrays: Record<string, PoseSeries> = {};

  if (!parentGroup) {
    return poseArrays;
  }

  for (const childName of parentGroup.keys()) {
    const childGroup = maybeChildGroup(parentGroup, childName);
    const dataset = childGroup ? maybeChildDataset(childGroup, 'pose') : null;
    if (!dataset || !isPoseShape(dataset.shape)) {
      continue;
    }

    const values = dataset.to_array();
    if (!Array.isArray(values)) {
      continue;
    }

    poseArrays[childName] = values as PoseSeries;
  }

  return poseArrays;
}

function mergePoseArrays(
  ...sources: Record<string, PoseSeries>[]
): Record<string, PoseSeries> {
  const merged: Record<string, PoseSeries> = {};
  for (const source of sources) {
    Object.assign(merged, source);
  }
  return merged;
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

function listDemos(h5File: H5WasmFile): DemoInfo[] {
  const dataGroup = getDataGroup(h5File);
  const demoNames = dataGroup
    .keys()
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

function safeJsonParse(value: string): unknown {
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

function coerceArticulationValue(value: unknown): unknown {
  if (typeof value !== 'string') {
    return value;
  }

  const trimmed = value.trim();
  if (trimmed === '') {
    return null;
  }

  if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
    const parsed = safeJsonParse(trimmed);
    if (parsed !== null) {
      return parsed;
    }
  }

  return value;
}

function parseIndexRange(
  value: unknown,
): { start: number; end: number } | null {
  const coerced = coerceArticulationValue(value);
  if (Array.isArray(coerced) && coerced.length === 2) {
    const start = optionalInt(coerced[0]);
    const end = optionalInt(coerced[1]);
    if (start != null && end != null && start <= end) {
      return { start, end };
    }
  }

  if (typeof coerced !== 'string') {
    return null;
  }

  const match = coerced.match(/^\s*\[\s*(-?\d+)\s*(?::|,)\s*(-?\d+)\s*\]\s*$/u);
  if (!match) {
    return null;
  }

  const start = Number.parseInt(match[1], 10);
  const end = Number.parseInt(match[2], 10);
  if (!Number.isInteger(start) || !Number.isInteger(end) || start > end) {
    return null;
  }

  return { start, end };
}

function parseJointIndices(
  articulationName: string,
  value: unknown,
): ArticulationJoint[] {
  const coerced = coerceArticulationValue(value);
  if (!Array.isArray(coerced)) {
    return [];
  }

  const joints: ArticulationJoint[] = [];
  for (const entry of coerced) {
    if (Array.isArray(entry) && entry.length >= 2) {
      const index = optionalInt(entry[1]);
      if (index == null) {
        continue;
      }
      joints.push({
        articulationName,
        name: String(entry[0] ?? `joint_${index}`),
        index,
      });
      continue;
    }

    if (!entry || typeof entry !== 'object') {
      continue;
    }

    const record = entry as Record<string, unknown>;
    const index = optionalInt(
      record.column_index ?? record.index ?? record.joint_index,
    );
    if (index == null) {
      continue;
    }

    const rawName = record.joint_name ?? record.name;
    joints.push({
      articulationName,
      name:
        typeof rawName === 'string' || typeof rawName === 'number'
          ? String(rawName)
          : `joint_${index}`,
      index,
    });
  }

  return joints.sort(
    (left, right) =>
      left.articulationName.localeCompare(right.articulationName) ||
      left.index - right.index ||
      left.name.localeCompare(right.name),
  );
}

function parsePoseOrder(value: unknown): string[] {
  const coerced = coerceArticulationValue(value);
  if (!Array.isArray(coerced)) {
    return [];
  }

  return coerced
    .filter((entry): entry is string => typeof entry === 'string')
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0);
}

function parseComponentSlices(
  articulationName: string,
  value: unknown,
  poseOrder: readonly string[] = [],
): ArticulationEndEffector[] {
  const coerced = coerceArticulationValue(value);
  if (!coerced || typeof coerced !== 'object' || Array.isArray(coerced)) {
    return [];
  }

  const record = coerced as Record<string, unknown>;
  const directPoseRange = parseIndexRange(record.pose);
  if (directPoseRange) {
    const gripperRange = parseIndexRange(record.gripper);
    return [
      {
        name: articulationName,
        poseStart: directPoseRange.start,
        poseEnd: directPoseRange.end,
        poseOrder: [...poseOrder],
        gripperStart: gripperRange?.start ?? null,
        gripperEnd: gripperRange?.end ?? null,
      },
    ];
  }

  const endEffectors: ArticulationEndEffector[] = [];
  for (const [eefName, eefValue] of Object.entries(record)) {
    if (!eefValue || typeof eefValue !== 'object' || Array.isArray(eefValue)) {
      continue;
    }

    const eefRecord = eefValue as Record<string, unknown>;
    const poseRange = parseIndexRange(eefRecord.pose);
    if (!poseRange) {
      continue;
    }

    const gripperRange = parseIndexRange(eefRecord.gripper);
    endEffectors.push({
      name: eefName,
      poseStart: poseRange.start,
      poseEnd: poseRange.end,
      poseOrder: [...poseOrder],
      gripperStart: gripperRange?.start ?? null,
      gripperEnd: gripperRange?.end ?? null,
    });
  }

  return endEffectors.sort((left, right) =>
    left.name.localeCompare(right.name),
  );
}

function readStandardArticulationsFromAttrs(dataGroup: H5WasmGroup): unknown {
  const articulationNames = [
    ...new Set(
      Object.keys(dataGroup.attrs)
        .map((key) => key.match(/^articulations\/([^/]+)\//u)?.[1])
        .filter((name): name is string => Boolean(name)),
    ),
  ].sort((left, right) => left.localeCompare(right));

  if (articulationNames.length === 0) {
    return null;
  }

  const joints: ArticulationJoint[] = [];
  const endEffectors: ArticulationEndEffector[] = [];
  let jointNumber = 0;
  let hasJointNumber = false;

  for (const articulationName of articulationNames) {
    const prefix = `articulations/${articulationName}`;
    const articulationJointNumber = optionalInt(
      getAttributeValue(dataGroup, `${prefix}/joint_number`),
    );
    if (articulationJointNumber != null) {
      jointNumber += articulationJointNumber;
      hasJointNumber = true;
    }

    joints.push(
      ...parseJointIndices(
        articulationName,
        getAttributeValue(dataGroup, `${prefix}/joints/joint_indices`),
      ),
    );
    const poseOrder = parsePoseOrder(
      getAttributeValue(dataGroup, `${prefix}/pose/pose_order`),
    );
    endEffectors.push(
      ...parseComponentSlices(
        articulationName,
        getAttributeValue(dataGroup, `${prefix}/pose/component_slices`),
        poseOrder,
      ),
    );
  }

  return {
    name: articulationNames.join(', '),
    joint_number: hasJointNumber ? jointNumber : null,
    joints,
    end_effectors: endEffectors,
  };
}

function readArticulationFromAttrs(dataGroup: H5WasmGroup): unknown {
  const standard = readStandardArticulationsFromAttrs(dataGroup);
  if (standard) {
    return standard;
  }

  const direct = getAttributeValue(dataGroup, 'articulation');
  if (direct != null) {
    return coerceArticulationValue(direct);
  }

  const flatKeys = [
    'articulation/name',
    'articulation/joint_number',
    'articulation/segmentation',
    'articulation/end_effectors',
  ];
  const hasFlat = flatKeys.some((key) => key in dataGroup.attrs);
  if (!hasFlat) {
    return null;
  }

  return {
    name: coerceArticulationValue(
      getAttributeValue(dataGroup, 'articulation/name'),
    ),
    joint_number: coerceArticulationValue(
      getAttributeValue(dataGroup, 'articulation/joint_number'),
    ),
    segmentation: coerceArticulationValue(
      getAttributeValue(dataGroup, 'articulation/segmentation'),
    ),
    end_effectors: coerceArticulationValue(
      getAttributeValue(dataGroup, 'articulation/end_effectors'),
    ),
  };
}

function readArticulationFromGroup(dataGroup: H5WasmGroup): unknown {
  const articulationGroup = maybeChildGroup(dataGroup, 'articulation');
  if (!articulationGroup) {
    return null;
  }

  const segmentationGroup = maybeChildGroup(articulationGroup, 'segmentation');
  const endEffectorsGroup = maybeChildGroup(articulationGroup, 'end_effectors');

  const segmentation: Record<string, { target: unknown; obs: unknown }> = {};
  if (segmentationGroup) {
    for (const segmentName of segmentationGroup.keys()) {
      const segmentChild = maybeChildGroup(segmentationGroup, segmentName);
      if (!segmentChild) {
        continue;
      }
      segmentation[segmentName] = {
        target: getAttributeValue(segmentChild, 'target'),
        obs: getAttributeValue(segmentChild, 'obs'),
      };
    }
  }

  const endEffectors: Record<string, { pose: unknown; gripper: unknown }> = {};
  if (endEffectorsGroup) {
    for (const eefName of endEffectorsGroup.keys()) {
      const eefChild = maybeChildGroup(endEffectorsGroup, eefName);
      if (!eefChild) {
        continue;
      }
      endEffectors[eefName] = {
        pose: getAttributeValue(eefChild, 'pose'),
        gripper: getAttributeValue(eefChild, 'gripper'),
      };
    }
  }

  return {
    name: getAttributeValue(articulationGroup, 'name'),
    joint_number: getAttributeValue(articulationGroup, 'joint_number'),
    segmentation,
    end_effectors: endEffectors,
  };
}

function parseInclusiveRange(
  value: unknown,
): { start: number; end: number } | null {
  return parseIndexRange(value);
}

function parseArticulation(raw: unknown): ParsedArticulation | null {
  if (!raw || typeof raw !== 'object') {
    return null;
  }

  const rawRecord = raw as Record<string, unknown>;
  const name = typeof rawRecord.name === 'string' ? rawRecord.name : '';
  const jointNumberRaw = rawRecord.joint_number;
  const jointNumber = optionalInt(jointNumberRaw);

  const segmentation: ArticulationSegment[] = [];
  const segRaw = rawRecord.segmentation;
  if (segRaw && typeof segRaw === 'object' && !Array.isArray(segRaw)) {
    for (const [segName, segValue] of Object.entries(
      segRaw as Record<string, unknown>,
    )) {
      if (!segValue || typeof segValue !== 'object') {
        continue;
      }
      const segRecord = segValue as Record<string, unknown>;
      const targetRange = parseInclusiveRange(segRecord.target);
      const obsRange = parseInclusiveRange(segRecord.obs);
      if (!targetRange || !obsRange) {
        continue;
      }
      segmentation.push({
        name: segName,
        targetStart: targetRange.start,
        targetEnd: targetRange.end,
        obsStart: obsRange.start,
        obsEnd: obsRange.end,
      });
    }
    segmentation.sort((left, right) => left.name.localeCompare(right.name));
  }

  const endEffectors: ArticulationEndEffector[] = [];
  const eefRaw = rawRecord.end_effectors;
  if (Array.isArray(eefRaw)) {
    for (const eefValue of eefRaw) {
      if (
        !eefValue ||
        typeof eefValue !== 'object' ||
        Array.isArray(eefValue)
      ) {
        continue;
      }
      const eefRecord = eefValue as Record<string, unknown>;
      const poseStart = optionalInt(eefRecord.poseStart);
      const poseEnd = optionalInt(eefRecord.poseEnd);
      if (
        typeof eefRecord.name !== 'string' ||
        poseStart == null ||
        poseEnd == null
      ) {
        continue;
      }

      const poseOrder = Array.isArray(eefRecord.poseOrder)
        ? eefRecord.poseOrder.filter(
            (entry): entry is string => typeof entry === 'string',
          )
        : [];
      endEffectors.push({
        name: eefRecord.name,
        poseStart,
        poseEnd,
        poseOrder,
        gripperStart: optionalInt(eefRecord.gripperStart),
        gripperEnd: optionalInt(eefRecord.gripperEnd),
      });
    }
    endEffectors.sort((left, right) => left.name.localeCompare(right.name));
  } else if (eefRaw && typeof eefRaw === 'object') {
    for (const [eefName, eefValue] of Object.entries(
      eefRaw as Record<string, unknown>,
    )) {
      if (!eefValue || typeof eefValue !== 'object') {
        continue;
      }
      const eefRecord = eefValue as Record<string, unknown>;
      const poseRange = parseInclusiveRange(eefRecord.pose);
      const gripperRange = parseInclusiveRange(eefRecord.gripper);
      if (!poseRange) {
        continue;
      }
      endEffectors.push({
        name: eefName,
        poseStart: poseRange.start,
        poseEnd: poseRange.end,
        poseOrder: [],
        gripperStart: gripperRange?.start ?? null,
        gripperEnd: gripperRange?.end ?? null,
      });
    }
    endEffectors.sort((left, right) => left.name.localeCompare(right.name));
  }

  const joints = Array.isArray(rawRecord.joints)
    ? (rawRecord.joints as ArticulationJoint[]).filter(
        (joint) =>
          typeof joint.articulationName === 'string' &&
          typeof joint.name === 'string' &&
          Number.isInteger(joint.index),
      )
    : [];

  if (
    !name &&
    segmentation.length === 0 &&
    endEffectors.length === 0 &&
    joints.length === 0
  ) {
    return null;
  }

  return {
    name,
    jointNumber,
    segmentation,
    endEffectors,
    joints,
  };
}

function readArticulation(h5File: H5WasmFile): ParsedArticulation | null {
  let dataGroup: H5WasmGroup;
  try {
    dataGroup = getDataGroup(h5File);
  } catch {
    return null;
  }

  const fromAttrs = readArticulationFromAttrs(dataGroup);
  const fromAttrsParsed = parseArticulation(fromAttrs);
  if (fromAttrsParsed) {
    return fromAttrsParsed;
  }

  return parseArticulation(readArticulationFromGroup(dataGroup));
}

function read2DDataset(
  demoGroup: H5WasmGroup,
  path: string,
): number[][] | null {
  const dataset = maybeChildDataset(demoGroup, path);
  if (!dataset || !Array.isArray(dataset.shape) || dataset.shape.length !== 2) {
    return null;
  }

  const [rowCount, colCount] = dataset.shape;
  if (
    !Number.isInteger(rowCount) ||
    !Number.isInteger(colCount) ||
    rowCount <= 0 ||
    colCount <= 0
  ) {
    return null;
  }

  const arrayValue = dataset.to_array();
  if (
    Array.isArray(arrayValue) &&
    arrayValue.length > 0 &&
    Array.isArray(arrayValue[0])
  ) {
    return arrayValue as number[][];
  }

  const flatSource = ArrayBuffer.isView(arrayValue)
    ? (arrayValue as ArrayLike<number>)
    : ArrayBuffer.isView(dataset.value)
      ? (dataset.value as ArrayLike<number>)
      : null;

  if (!flatSource || flatSource.length < rowCount * colCount) {
    return null;
  }

  const rows: number[][] = new Array(rowCount);
  for (let rowIdx = 0; rowIdx < rowCount; rowIdx += 1) {
    const offset = rowIdx * colCount;
    const row: number[] = new Array(colCount);
    for (let colIdx = 0; colIdx < colCount; colIdx += 1) {
      row[colIdx] = Number(flatSource[offset + colIdx]);
    }
    rows[rowIdx] = row;
  }
  return rows;
}

const OBS_ARTICULATION_GROUP_NAMES = ['articulations', 'articulation'] as const;

function findObsJointPositionPath(
  demoGroup: H5WasmGroup,
  articulationName: string,
): string | null {
  const obsGroup = maybeChildGroup(demoGroup, 'obs');
  if (!obsGroup) {
    return null;
  }

  for (const containerName of OBS_ARTICULATION_GROUP_NAMES) {
    const articulationGroup = maybeChildGroup(obsGroup, containerName);
    if (!articulationGroup) {
      continue;
    }

    if (articulationName) {
      const directChild = maybeChildGroup(articulationGroup, articulationName);
      if (directChild && maybeChildDataset(directChild, 'joint_position')) {
        return `obs/${containerName}/${articulationName}/joint_position`;
      }
    }

    for (const childName of articulationGroup.keys()) {
      const child = maybeChildGroup(articulationGroup, childName);
      if (child && maybeChildDataset(child, 'joint_position')) {
        return `obs/${containerName}/${childName}/joint_position`;
      }
    }
  }

  return null;
}

function extractColumnAtStep(
  data: number[][] | null,
  stepIdx: number,
  column: number,
): number | null {
  if (!data || column < 0) {
    return null;
  }

  const row = data[stepIdx];
  if (!Array.isArray(row) || column >= row.length) {
    return null;
  }

  const value = row[column];
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return null;
  }

  return roundFloat(value);
}

function poseComponentColumn(
  eef: ArticulationEndEffector,
  component: 'x' | 'y' | 'z',
): number {
  const fallbackOffset = component === 'x' ? 0 : component === 'y' ? 1 : 2;
  const poseOrderOffset = eef.poseOrder.indexOf(component);
  const offset = poseOrderOffset >= 0 ? poseOrderOffset : fallbackOffset;
  const column = eef.poseStart + offset;
  return column < eef.poseEnd ? column : -1;
}

function makeJointRowKey(
  articulationName: string,
  jointName: string,
  jointIndex: number,
): string {
  return `${articulationName}::${jointName}::${jointIndex}`;
}

function loadEndEffectorPoses(
  demoGroup: H5WasmGroup,
): Record<string, PoseSeries> {
  // New standard schema: `obs/end_effectors/<name>/pose` (T, 4, 4).
  const obsGroup = maybeChildGroup(demoGroup, 'obs');
  const endEffectorsGroup = obsGroup
    ? maybeChildGroup(obsGroup, 'end_effectors')
    : null;
  const obsEefPoseGroup = obsGroup
    ? maybeChildGroup(obsGroup, 'eef_pose')
    : null;

  // Legacy schema: `obs/eef_pose/<name>` (T, 4, 4) or `datagen_info/eef_pose/<name>`.
  const legacyGroup = findPoseGroup(demoGroup, 'eef_pose');
  return mergePoseArrays(
    loadPoseArrays(legacyGroup),
    loadPoseArrays(obsEefPoseGroup),
    loadNestedPoseArrays(endEffectorsGroup),
  );
}

function loadObjectKeypointPoses(
  demoGroup: H5WasmGroup,
): Record<string, PoseSeries> {
  // New schema: `obs/objects/<name>/pose` (T, 4, 4).
  const obsGroup = maybeChildGroup(demoGroup, 'obs');
  const objectsGroup = obsGroup ? maybeChildGroup(obsGroup, 'objects') : null;

  // New schema (datagen): `obs/datagen_info/object_pose/<name>` or legacy `obs/object_pose/<name>`.
  const legacyGroup = findPoseGroup(demoGroup, 'object_pose');
  return mergePoseArrays(
    loadPoseArrays(legacyGroup),
    loadNestedPoseArrays(objectsGroup),
  );
}

function buildDemoRows(entry: OpenSourceEntry, demoName: string): DemoRow[] {
  const demoGroup = getDemoGroup(entry, demoName);
  const articulation = entry.articulation;

  const eefPosePostStepGroup = findPoseGroup(demoGroup, 'eef_pose_post_step');
  const eefPose = loadEndEffectorPoses(demoGroup);
  const eefPosePostStep = loadPoseArrays(eefPosePostStepGroup);
  const objectPose = loadObjectKeypointPoses(demoGroup);

  const usesActionsPoseTarget = (articulation?.endEffectors.length ?? 0) > 0;
  const actionsPose = usesActionsPoseTarget
    ? read2DDataset(demoGroup, 'actions/pose')
    : null;

  let legacyTargetEefPose: Record<string, PoseSeries> = {};
  if (!usesActionsPoseTarget) {
    const legacyTargetGroup =
      findPoseGroup(demoGroup, 'target_eef_pose') ??
      findPoseGroup(demoGroup, 'ik_input_eef_pose');
    legacyTargetEefPose = loadPoseArrays(legacyTargetGroup);
  }

  const articulationName = articulation?.name?.trim() ?? '';
  const segments = articulation?.segmentation ?? [];
  const joints = articulation?.joints ?? [];
  const actionsJoints =
    segments.length > 0 || joints.length > 0
      ? read2DDataset(demoGroup, 'actions/joints')
      : null;
  const obsJointPositionPath =
    segments.length > 0
      ? findObsJointPositionPath(demoGroup, articulationName)
      : null;
  const obsJointPosition = obsJointPositionPath
    ? read2DDataset(demoGroup, obsJointPositionPath)
    : null;
  const obsJointPositionByArticulation: Record<string, number[][] | null> = {};
  for (const jointArticulationName of [
    ...new Set(joints.map((joint) => joint.articulationName)),
  ]) {
    const path = findObsJointPositionPath(demoGroup, jointArticulationName);
    obsJointPositionByArticulation[jointArticulationName] = path
      ? read2DDataset(demoGroup, path)
      : null;
  }
  const hasMappedObsJointPosition = Object.values(
    obsJointPositionByArticulation,
  ).some((data) => data != null);

  if (
    Object.keys(eefPose).length === 0 &&
    Object.keys(legacyTargetEefPose).length === 0 &&
    Object.keys(eefPosePostStep).length === 0 &&
    Object.keys(objectPose).length === 0 &&
    !actionsPose &&
    !actionsJoints &&
    !obsJointPosition &&
    !hasMappedObsJointPosition
  ) {
    throw new Error(
      `Demo '${demoName}' does not contain usable pose datasets.`,
    );
  }

  const poseArrays = [
    ...Object.values(eefPose),
    ...Object.values(legacyTargetEefPose),
    ...Object.values(eefPosePostStep),
    ...Object.values(objectPose),
  ];
  const poseLengths = poseArrays
    .filter((series) =>
      series.every((frame) => Array.isArray(frame) && frame.length === 4),
    )
    .map((series) => series.length);

  const arrayLengths: number[] = [...poseLengths];
  if (actionsPose) arrayLengths.push(actionsPose.length);
  if (actionsJoints) arrayLengths.push(actionsJoints.length);
  if (obsJointPosition) arrayLengths.push(obsJointPosition.length);
  for (const mappedObsJointPosition of Object.values(
    obsJointPositionByArticulation,
  )) {
    if (mappedObsJointPosition)
      arrayLengths.push(mappedObsJointPosition.length);
  }

  if (arrayLengths.length === 0) {
    throw new Error(
      `Demo '${demoName}' does not contain valid pose or joint data.`,
    );
  }

  const numSteps = Math.max(...arrayLengths);
  const storedNumSamples = optionalInt(
    getAttributeValue(demoGroup, 'num_samples'),
  );
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

    for (const eefName of Object.keys(eefPose).sort((left, right) =>
      left.localeCompare(right),
    )) {
      const xyz = extractXYZAtStep(eefPose[eefName], stepIdx);
      row[`eef_${eefName}_x`] = xyz?.[0] ?? null;
      row[`eef_${eefName}_y`] = xyz?.[1] ?? null;
      row[`eef_${eefName}_z`] = xyz?.[2] ?? null;
    }

    if (usesActionsPoseTarget && articulation) {
      for (const eef of articulation.endEffectors) {
        const targetX = extractColumnAtStep(
          actionsPose,
          stepIdx,
          poseComponentColumn(eef, 'x'),
        );
        const targetY = extractColumnAtStep(
          actionsPose,
          stepIdx,
          poseComponentColumn(eef, 'y'),
        );
        const targetZ = extractColumnAtStep(
          actionsPose,
          stepIdx,
          poseComponentColumn(eef, 'z'),
        );
        row[`target_eef_${eef.name}_x`] = targetX;
        row[`target_eef_${eef.name}_y`] = targetY;
        row[`target_eef_${eef.name}_z`] = targetZ;
        row[`ik_input_eef_${eef.name}_x`] = targetX;
        row[`ik_input_eef_${eef.name}_y`] = targetY;
        row[`ik_input_eef_${eef.name}_z`] = targetZ;
      }
    } else {
      for (const eefName of Object.keys(legacyTargetEefPose).sort(
        (left, right) => left.localeCompare(right),
      )) {
        const targetXYZ = extractXYZAtStep(
          legacyTargetEefPose[eefName],
          stepIdx,
        );
        row[`target_eef_${eefName}_x`] = targetXYZ?.[0] ?? null;
        row[`target_eef_${eefName}_y`] = targetXYZ?.[1] ?? null;
        row[`target_eef_${eefName}_z`] = targetXYZ?.[2] ?? null;
        row[`ik_input_eef_${eefName}_x`] = targetXYZ?.[0] ?? null;
        row[`ik_input_eef_${eefName}_y`] = targetXYZ?.[1] ?? null;
        row[`ik_input_eef_${eefName}_z`] = targetXYZ?.[2] ?? null;
      }
    }

    for (const eefName of Object.keys(eefPosePostStep).sort((left, right) =>
      left.localeCompare(right),
    )) {
      const postStepXYZ = extractXYZAtStep(eefPosePostStep[eefName], stepIdx);
      row[`eef_post_step_${eefName}_x`] = postStepXYZ?.[0] ?? null;
      row[`eef_post_step_${eefName}_y`] = postStepXYZ?.[1] ?? null;
      row[`eef_post_step_${eefName}_z`] = postStepXYZ?.[2] ?? null;
    }

    for (const objectName of Object.keys(objectPose).sort((left, right) =>
      left.localeCompare(right),
    )) {
      const xyz = extractXYZAtStep(objectPose[objectName], stepIdx);
      row[`object_${objectName}_x`] = xyz?.[0] ?? null;
      row[`object_${objectName}_y`] = xyz?.[1] ?? null;
      row[`object_${objectName}_z`] = xyz?.[2] ?? null;
    }

    for (const segment of segments) {
      const jointCount = Math.min(
        segment.targetEnd - segment.targetStart + 1,
        segment.obsEnd - segment.obsStart + 1,
      );
      for (let offset = 0; offset < jointCount; offset += 1) {
        const targetCol = segment.targetStart + offset;
        const obsCol = segment.obsStart + offset;
        row[`joint_target_${segment.name}_${targetCol}`] = extractColumnAtStep(
          actionsJoints,
          stepIdx,
          targetCol,
        );
        row[`joint_obs_${segment.name}_${targetCol}`] = extractColumnAtStep(
          obsJointPosition,
          stepIdx,
          obsCol,
        );
      }
    }

    for (const joint of joints) {
      const rowKey = makeJointRowKey(
        joint.articulationName,
        joint.name,
        joint.index,
      );
      row[`joint_target_${rowKey}`] = extractColumnAtStep(
        actionsJoints,
        stepIdx,
        joint.index,
      );
      row[`joint_obs_${rowKey}`] = extractColumnAtStep(
        obsJointPositionByArticulation[joint.articulationName] ?? null,
        stepIdx,
        joint.index,
      );
    }

    rows.push(row);
  }

  return rows;
}

function readDatasetArray(dataset: H5WasmDataset): unknown {
  const arrayValue = dataset.to_array();
  if (arrayValue != null) {
    return arrayValue;
  }

  return dataset.json_value ?? dataset.value ?? null;
}

function formatDatasetDtype(dataset: H5WasmDataset): string {
  const dtype = dataset.dtype as unknown;
  if (typeof dtype === 'string') {
    return dtype;
  }

  try {
    return JSON.stringify(dtype);
  } catch {
    return String(dtype);
  }
}

function toComparisonValue(value: unknown): unknown {
  if (value == null) {
    return null;
  }

  if (typeof value === 'bigint') {
    return value.toString();
  }

  if (typeof value === 'number') {
    return Number.isFinite(value) ? value : String(value);
  }

  if (typeof value === 'string' || typeof value === 'boolean') {
    return value;
  }

  if (ArrayBuffer.isView(value)) {
    if ('length' in value) {
      return Array.from(value as unknown as ArrayLike<unknown>).map(
        toComparisonValue,
      );
    }

    return Array.from(
      new Uint8Array(value.buffer, value.byteOffset, value.byteLength),
    );
  }

  if (Array.isArray(value)) {
    return value.map(toComparisonValue);
  }

  if (value instanceof Map) {
    return Object.fromEntries(
      [...value.entries()].map(([key, entryValue]) => [
        String(key),
        toComparisonValue(entryValue),
      ]),
    );
  }

  if (typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(
        ([key, entryValue]) => [key, toComparisonValue(entryValue)],
      ),
    );
  }

  return String(value);
}

function comparisonSelectionForShape(shape: readonly number[] | null): {
  selection: string | null;
  slices: Array<[number, number]> | null;
} {
  if (!shape) {
    return { selection: null, slices: null };
  }

  if (shape.length === 0) {
    return { selection: 'scalar', slices: [] };
  }

  if (shape.length <= 2) {
    return {
      selection: shape.map(() => ':').join(','),
      slices: shape.map((size) => [0, size]),
    };
  }

  return {
    selection: shape
      .map((_, index) => (index < shape.length - 2 ? '0' : ':'))
      .join(','),
    slices: shape.map((size, index) =>
      index < shape.length - 2 ? [0, 1] : [0, size],
    ),
  };
}

function squeezeLeadingDimensions(value: unknown, count: number): unknown {
  let result = value;
  for (let index = 0; index < count; index += 1) {
    if (!Array.isArray(result)) {
      return result;
    }

    result = result[0];
  }

  return result;
}

function readDatasetComparisonValue(dataset: H5WasmDataset): {
  value: unknown;
  selection: string | null;
} {
  const shape = dataset.shape;
  const { selection, slices } = comparisonSelectionForShape(shape);

  if (shape && shape.length > 2 && slices) {
    const rawValue = dataset.slice(slices);
    return {
      value: toComparisonValue(
        squeezeLeadingDimensions(rawValue, shape.length - 2),
      ),
      selection,
    };
  }

  return {
    value: toComparisonValue(readDatasetArray(dataset)),
    selection,
  };
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function asNumberVector(value: unknown): number[] | null {
  if (!Array.isArray(value) || value.length === 0) {
    return null;
  }

  if (value.every(isFiniteNumber)) {
    return value as number[];
  }

  const firstRow = value[0];
  if (Array.isArray(firstRow) && firstRow.every(isFiniteNumber)) {
    return firstRow as number[];
  }

  return null;
}

function asPoseMatrix(value: unknown): number[][] | null {
  if (!Array.isArray(value) || value.length === 0) {
    return null;
  }

  const firstItem = value[0];
  if (Array.isArray(firstItem) && Array.isArray(firstItem[0])) {
    return asPoseMatrix(firstItem);
  }

  if (!value.every((row) => Array.isArray(row) && row.every(isFiniteNumber))) {
    return null;
  }

  return value as number[][];
}

function homogeneousMatrixToPoseVector(matrix: number[][]): number[] | null {
  if (matrix.length < 3) {
    return null;
  }
  const row0 = matrix[0];
  const row1 = matrix[1];
  const row2 = matrix[2];
  if (row0.length < 4 || row1.length < 4 || row2.length < 4) {
    return null;
  }

  const x = row0[3];
  const y = row1[3];
  const z = row2[3];

  const r00 = row0[0];
  const r01 = row0[1];
  const r02 = row0[2];
  const r10 = row1[0];
  const r11 = row1[1];
  const r12 = row1[2];
  const r20 = row2[0];
  const r21 = row2[1];
  const r22 = row2[2];

  // Shepperd's method: numerically stable rotation matrix → quaternion (wxyz).
  const trace = r00 + r11 + r22;
  let qw: number;
  let qx: number;
  let qy: number;
  let qz: number;
  if (trace > 0) {
    const s = Math.sqrt(trace + 1) * 2;
    qw = 0.25 * s;
    qx = (r21 - r12) / s;
    qy = (r02 - r20) / s;
    qz = (r10 - r01) / s;
  } else if (r00 > r11 && r00 > r22) {
    const s = Math.sqrt(1 + r00 - r11 - r22) * 2;
    qw = (r21 - r12) / s;
    qx = 0.25 * s;
    qy = (r01 + r10) / s;
    qz = (r02 + r20) / s;
  } else if (r11 > r22) {
    const s = Math.sqrt(1 + r11 - r00 - r22) * 2;
    qw = (r02 - r20) / s;
    qx = (r01 + r10) / s;
    qy = 0.25 * s;
    qz = (r12 + r21) / s;
  } else {
    const s = Math.sqrt(1 + r22 - r00 - r11) * 2;
    qw = (r10 - r01) / s;
    qx = (r02 + r20) / s;
    qy = (r12 + r21) / s;
    qz = 0.25 * s;
  }

  if (![x, y, z, qw, qx, qy, qz].every(isFiniteNumber)) {
    return null;
  }
  return [x, y, z, qw, qx, qy, qz];
}

// Current schema: `initial_state/rigid_objects/<name>/initial_pose`. Older
// variants live under `initial_state/objects/<name>/`.
function maybeObjectsRoot(demoGroup: H5WasmGroup): H5WasmGroup | null {
  return (
    maybeChildGroup(demoGroup, 'initial_state/rigid_objects') ??
    maybeChildGroup(demoGroup, 'initial_state/objects')
  );
}

// The `initial_pose` dataset is either a (1, 7) xyz+quat vector or a
// (1, 4, 4) / (4, 4) homogeneous transform matrix depending on how the file
// was authored. Returns an xyz+quat (wxyz) pose vector.
function parseObjectInitialPose(objectGroup: H5WasmGroup): number[] | null {
  const initialPose = maybeChildDataset(objectGroup, 'initial_pose');
  if (!initialPose) {
    return null;
  }

  const raw = readDatasetArray(initialPose);

  const vector = asNumberVector(raw);
  if (vector && vector.length >= 2) {
    return vector;
  }

  const matrix = asPoseMatrix(raw);
  if (matrix) {
    return homogeneousMatrixToPoseVector(matrix);
  }

  return null;
}

// Names of every rigid object that exposes a readable `initial_pose`, in the
// order they appear in the file.
function listRigidObjectNames(demoGroup: H5WasmGroup): string[] {
  const objectsRoot = maybeObjectsRoot(demoGroup);
  if (!objectsRoot) {
    return [];
  }

  const names: string[] = [];
  for (const objectName of objectsRoot.keys()) {
    const objectGroup = maybeChildGroup(objectsRoot, objectName);
    if (objectGroup && maybeChildDataset(objectGroup, 'initial_pose')) {
      names.push(objectName);
    }
  }

  return names;
}

function readInitialPose(
  demoGroup: H5WasmGroup,
  objectName: string | null = null,
): number[] | null {
  const objectsRoot = maybeObjectsRoot(demoGroup);
  if (!objectsRoot) {
    return null;
  }

  // Explicit object selection: read only the requested object.
  if (objectName) {
    const objectGroup = maybeChildGroup(objectsRoot, objectName);
    return objectGroup ? parseObjectInitialPose(objectGroup) : null;
  }

  // Default: first object with a parseable initial_pose.
  for (const name of objectsRoot.keys()) {
    const objectGroup = maybeChildGroup(objectsRoot, name);
    if (!objectGroup) {
      continue;
    }

    const pose = parseObjectInitialPose(objectGroup);
    if (pose) {
      return pose;
    }
  }

  return null;
}

function readAnchorXY(
  demoGroup: H5WasmGroup,
  _anchor: ObjectDistributionAnchor,
  objectName: string | null = null,
): [number, number] | null {
  const pose = readInitialPose(demoGroup, objectName);
  const x = pose?.[0];
  const y = pose?.[1];
  if (!isFiniteNumber(x) || !isFiniteNumber(y)) {
    return null;
  }

  return [roundFloat(x), roundFloat(y)];
}

function readSourceDemoIndices(
  demoGroup: H5WasmGroup,
  datasetPath: string,
): number[] {
  const dataset = maybeChildDataset(demoGroup, datasetPath);
  if (!dataset) {
    return [];
  }

  const value = dataset.value;
  if (ArrayBuffer.isView(value)) {
    return [...value]
      .map((entry) => Math.trunc(Number(entry)))
      .filter(Number.isFinite);
  }

  const arrayValue = readDatasetArray(dataset);
  if (!Array.isArray(arrayValue)) {
    return [];
  }

  return arrayValue
    .flat(Infinity)
    .map((entry) => Math.trunc(Number(entry)))
    .filter(Number.isFinite);
}

function episodeLength(demoGroup: H5WasmGroup): number | null {
  // New schema: `actions/pose` and `actions/joints` both have a leading T dim.
  const actionsJoints = maybeChildDataset(demoGroup, 'actions/joints');
  if (actionsJoints?.shape?.[0] != null) {
    return actionsJoints.shape[0];
  }

  const actionsPose = maybeChildDataset(demoGroup, 'actions/pose');
  if (actionsPose?.shape?.[0] != null) {
    return actionsPose.shape[0];
  }

  // Legacy schema: a single `actions` dataset.
  const actions = maybeChildDataset(demoGroup, 'actions');
  if (actions?.shape?.[0] != null) {
    return actions.shape[0];
  }

  return optionalInt(getAttributeValue(demoGroup, 'num_samples'));
}

function makeTeleopId(entry: OpenSourceEntry, demoName: string): string {
  return `${entry.datasetName}::${demoName}`;
}

function collectTeleopSources(
  entry: OpenSourceEntry,
  anchor: ObjectDistributionAnchor,
  objectName: string | null,
): {
  points: ObjectDistributionPoint[];
  byDemoName: Record<string, TeleopSource[]>;
  diagnostics: ObjectDistributionSourceDiagnostics;
} {
  const points: ObjectDistributionPoint[] = [];
  const byDemoName: Record<string, TeleopSource[]> = {};
  let missingAnchorCount = 0;

  for (const demo of entry.demos) {
    const demoGroup = getDemoGroup(entry, demo.name);
    const xy = readAnchorXY(demoGroup, anchor, objectName);
    if (!xy) {
      missingAnchorCount += 1;
      continue;
    }

    // Every teleop demo is registered as a candidate source, anchored by its
    // initial-pose XY. Generated demos link back to these by the explicit
    // `reference_demo_indices`, so the source-arrow target is this position.
    const initialPose = readInitialPose(demoGroup, objectName);
    const teleopSource: TeleopSource = {
      teleopId: makeTeleopId(entry, demo.name),
      datasetName: entry.datasetName,
      demoName: demo.name,
      x: xy[0],
      y: xy[1],
    };
    byDemoName[demo.name] = [...(byDemoName[demo.name] ?? []), teleopSource];
    points.push({
      category: 'teleop',
      datasetName: entry.datasetName,
      demoName: demo.name,
      x: xy[0],
      y: xy[1],
      initialX: initialPose?.[0] ?? null,
      initialY: initialPose?.[1] ?? null,
      initialRx: initialPose?.[3] ?? null,
      initialRy: initialPose?.[4] ?? null,
      numSamples: episodeLength(demoGroup),
      sourceLeft: '-',
      sourceRight: '-',
      sourceLeftDetails: [],
      sourceRightDetails: [],
    });
  }

  return {
    points,
    byDemoName,
    diagnostics: {
      totalDemos: entry.demos.length,
      includedDemos: points.length,
      missingAnchorCount,
    },
  };
}

// Resolve the source-demo arrows for one arm. The current schema gives the
// source demo index explicitly per subtask slot in `reference_demo_indices`, so
// each referenced index simply links to that teleop demo's anchor position. We
// group repeated references to the same demo into a single arrow.
function resolveSideSourceDetails(
  selectedIndices: readonly number[],
  side: 'left' | 'right',
  teleopSourcesByDemo: Record<string, TeleopSource[]>,
): {
  rawSource: string;
  details: ObjectDistributionSourceDetail[];
} {
  const values = [...selectedIndices]
    .map((value) => Math.trunc(value))
    .filter(Number.isFinite);
  const rawSource = values.length > 0 ? values.join(',') : '-';
  const grouped = new Map<
    string,
    {
      teleopId: string;
      datasetName: string;
      demoName: string;
      x: number;
      y: number;
      slots: number[];
    }
  >();

  values.forEach((indexValue, slot) => {
    const demoName = `demo_${indexValue}`;
    const source = teleopSourcesByDemo[demoName]?.[0];
    if (!source) {
      return;
    }

    const existing = grouped.get(source.teleopId) ?? {
      teleopId: source.teleopId,
      datasetName: source.datasetName,
      demoName: source.demoName,
      x: source.x,
      y: source.y,
      slots: [],
    };
    existing.slots.push(slot);
    grouped.set(source.teleopId, existing);
  });

  const details = [...grouped.values()]
    .sort((left, right) => left.demoName.localeCompare(right.demoName))
    .map(
      (record) =>
        ({
          teleopId: record.teleopId,
          datasetName: record.datasetName,
          demoName: record.demoName,
          x: record.x,
          y: record.y,
          slots: [...record.slots].sort((left, right) => left - right),
          textLabel: record.demoName,
          hoverLabel: `${side.toUpperCase()} arm → ${record.demoName}`,
        }) satisfies ObjectDistributionSourceDetail,
    );

  return { rawSource, details };
}

function collectGeneratedObjectPoints(
  entry: OpenSourceEntry,
  category: Exclude<ObjectDistributionCategory, 'teleop'>,
  anchor: ObjectDistributionAnchor,
  teleopSourcesByDemo: Record<string, TeleopSource[]>,
  objectName: string | null,
): ObjectDistributionPoint[] {
  const points: ObjectDistributionPoint[] = [];

  for (const demo of entry.demos) {
    const demoGroup = getDemoGroup(entry, demo.name);
    const initialPose = readInitialPose(demoGroup, objectName);
    const anchorXY = readAnchorXY(demoGroup, anchor, objectName);
    let xy: [number, number] | null = anchorXY;
    if (
      !xy &&
      initialPose &&
      isFiniteNumber(initialPose[0]) &&
      isFiniteNumber(initialPose[1])
    ) {
      xy = [roundFloat(initialPose[0]), roundFloat(initialPose[1])];
    }
    if (!xy) {
      continue;
    }

    // New schema uses `reference_demo_indices/<articulation_name>`; legacy used `source_demo_indices`.
    // The articulation name varies across tasks (`left`/`right` or `left_arm`/`right_arm`).
    const leftIndices = [
      ...readSourceDemoIndices(demoGroup, 'reference_demo_indices/left_arm'),
      ...readSourceDemoIndices(demoGroup, 'reference_demo_indices/left'),
      ...readSourceDemoIndices(demoGroup, 'source_demo_indices/left_arm'),
      ...readSourceDemoIndices(demoGroup, 'source_demo_indices/left'),
    ];
    const rightIndices = [
      ...readSourceDemoIndices(demoGroup, 'reference_demo_indices/right_arm'),
      ...readSourceDemoIndices(demoGroup, 'reference_demo_indices/right'),
      ...readSourceDemoIndices(demoGroup, 'source_demo_indices/right_arm'),
      ...readSourceDemoIndices(demoGroup, 'source_demo_indices/right'),
    ];
    const leftSource = resolveSideSourceDetails(
      leftIndices,
      'left',
      teleopSourcesByDemo,
    );
    const rightSource = resolveSideSourceDetails(
      rightIndices,
      'right',
      teleopSourcesByDemo,
    );

    points.push({
      category,
      datasetName: entry.datasetName,
      demoName: demo.name,
      x: xy[0],
      y: xy[1],
      initialX: initialPose?.[0] ?? null,
      initialY: initialPose?.[1] ?? null,
      initialRx: initialPose?.[3] ?? null,
      initialRy: initialPose?.[4] ?? null,
      numSamples: episodeLength(demoGroup),
      sourceLeft: leftSource.rawSource,
      sourceRight: rightSource.rawSource,
      sourceLeftDetails: leftSource.details,
      sourceRightDetails: rightSource.details,
    });
  }

  return points;
}

function loadObjectDistribution(
  request: ObjectDistributionRequest,
): ObjectDistributionResult {
  if (
    !(OBJECT_DISTRIBUTION_ANCHORS as readonly string[]).includes(request.anchor)
  ) {
    throw new Error(
      `Unsupported object distribution anchor: ${request.anchor}`,
    );
  }

  const successEntry = request.successSourceId
    ? (openSources.get(request.successSourceId) ?? null)
    : null;
  const failedEntry = request.failedSourceId
    ? (openSources.get(request.failedSourceId) ?? null)
    : null;
  const teleopEntry = request.teleopSourceId
    ? (openSources.get(request.teleopSourceId) ?? null)
    : null;

  if (request.successSourceId && !successEntry) {
    throw new Error(
      'Selected successful generated source is no longer available.',
    );
  }
  if (request.failedSourceId && !failedEntry) {
    throw new Error('Selected failed generated source is no longer available.');
  }
  if (request.teleopSourceId && !teleopEntry) {
    throw new Error('Selected teleop source is no longer available.');
  }

  const objectName = request.objectName ?? null;
  const availableObjects = collectAvailableObjects([
    successEntry,
    failedEntry,
    teleopEntry,
  ]);

  const teleopCollection = teleopEntry
    ? collectTeleopSources(teleopEntry, request.anchor, objectName)
    : { points: [], byDemoName: {}, diagnostics: null };

  return {
    anchor: request.anchor,
    successPoints: successEntry
      ? collectGeneratedObjectPoints(
          successEntry,
          'success',
          request.anchor,
          teleopCollection.byDemoName,
          objectName,
        )
      : [],
    failedPoints: failedEntry
      ? collectGeneratedObjectPoints(
          failedEntry,
          'failed',
          request.anchor,
          teleopCollection.byDemoName,
          objectName,
        )
      : [],
    teleopPoints: teleopCollection.points,
    teleopDiagnostics: teleopCollection.diagnostics,
    availableObjects,
  };
}

// Union of rigid-object names across the selected datasets, preserving the
// in-file order of the first dataset that exposes them. Object sets are fixed
// per scene, so inspecting the first demo of each entry is sufficient.
function collectAvailableObjects(
  entries: (OpenSourceEntry | null)[],
): string[] {
  const names: string[] = [];
  const seen = new Set<string>();

  for (const entry of entries) {
    if (!entry || entry.demos.length === 0) {
      continue;
    }

    const demoGroup = getDemoGroup(entry, entry.demos[0].name);
    for (const name of listRigidObjectNames(demoGroup)) {
      if (!seen.has(name)) {
        seen.add(name);
        names.push(name);
      }
    }
  }

  return names;
}

interface H5AttributeOwner {
  path: string;
  attrs: Record<
    string,
    {
      value: unknown;
      json_value: unknown;
      shape: number[] | null;
      dtype: Parameters<H5WasmGroup['create_attribute']>[3];
    }
  >;
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

function listDatasetProcessingInfo(
  entry: OpenSourceEntry,
): DatasetProcessingSourceInfo {
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

function demoHasPoseTraceData(entry: OpenSourceEntry, demoName: string) {
  try {
    return buildDemoRows(entry, demoName).length > 0;
  } catch {
    return false;
  }
}

function demoHasObjectDistributionData(
  entry: OpenSourceEntry,
  demoName: string,
) {
  try {
    const demoGroup = getDemoGroup(entry, demoName);
    return (
      readAnchorXY(demoGroup, 'initial_pose', null) !== null ||
      readInitialPose(demoGroup, null) !== null
    );
  } catch {
    return false;
  }
}

function inspectSourceFeatures(
  entry: OpenSourceEntry,
): SourceFeatureCapabilities {
  const info = listDatasetProcessingInfo(entry);
  let poseTraceDemoCount = 0;
  let videoDemoCount = 0;
  let videoCount = 0;
  let objectDistributionDemoCount = 0;

  for (const demo of entry.demos) {
    if (poseTraceDemoCount === 0 && demoHasPoseTraceData(entry, demo.name)) {
      poseTraceDemoCount = 1;
    }

    if (
      objectDistributionDemoCount === 0 &&
      demoHasObjectDistributionData(entry, demo.name)
    ) {
      objectDistributionDemoCount = 1;
    }

    if (videoDemoCount === 0) {
      const videos = listDemoVideoInfo(entry, demo.name);
      if (videos.length > 0) {
        videoDemoCount = 1;
        videoCount = videos.length;
      }
    }

    if (
      poseTraceDemoCount > 0 &&
      videoDemoCount > 0 &&
      objectDistributionDemoCount > 0
    ) {
      break;
    }
  }

  return {
    demoCount: entry.demos.length,
    keyCount: info.keyPaths.length,
    poseTraceDemoCount,
    videoDemoCount,
    videoCount,
    objectDistributionDemoCount,
  };
}

function loadDatasetComparisonValues(
  entry: OpenSourceEntry,
  demoName: string,
  keyPaths: readonly string[],
): DatasetComparisonValuesResult {
  const demoGroup = getDemoGroup(entry, demoName);
  const uniqueKeyPaths = [...new Set(keyPaths)].sort((left, right) =>
    left.localeCompare(right),
  );

  return {
    demoName,
    values: uniqueKeyPaths.map<DatasetComparisonValue>((keyPath) => {
      try {
        const dataset = maybeChildDataset(demoGroup, keyPath);
        if (!dataset) {
          return {
            keyPath,
            status: 'missing',
            shape: null,
            dtype: null,
            selection: null,
            value: null,
            error: `Dataset '${keyPath}' is not present in ${demoName}.`,
          };
        }

        const { value, selection } = readDatasetComparisonValue(dataset);

        return {
          keyPath,
          status: 'ok',
          shape: dataset.shape ? [...dataset.shape] : null,
          dtype: formatDatasetDtype(dataset),
          selection,
          value,
        };
      } catch (error: unknown) {
        return {
          keyPath,
          status: 'error',
          shape: null,
          dtype: null,
          selection: null,
          value: null,
          error: error instanceof Error ? error.message : String(error),
        };
      }
    }),
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
      throw new Error(
        'One of the selected source datasets is no longer available.',
      );
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

function buildDatasetCopyPlan(
  selectedKeys: readonly string[],
): DatasetCopyPlanGroup[] {
  const planByParentGroup = new Map<string, DatasetCopyPlanGroup>();

  for (const keyPath of selectedKeys) {
    const splitIndex = keyPath.lastIndexOf('/');
    const parentGroupPath = splitIndex >= 0 ? keyPath.slice(0, splitIndex) : '';
    const datasetName =
      splitIndex >= 0 ? keyPath.slice(splitIndex + 1) : keyPath;

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

function getDatasetCompressionConfig(
  dataset: H5WasmDataset,
): H5WasmCompressionConfig {
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
  for (
    let offset = firstBatchSize;
    offset < totalRows;
    offset += CHUNKED_COPY_ROW_BATCH
  ) {
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

type SliceProgressCallback = (
  datasetPath: string,
  copiedRows: number,
  totalRows: number,
) => void;

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

      if (
        estimatedBytes > CHUNKED_COPY_BYTE_THRESHOLD &&
        metadata.shape &&
        metadata.shape.length > 0 &&
        metadata.chunks
      ) {
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
  const selectedKeys = [...new Set(request.selectedKeys)].sort((left, right) =>
    left.localeCompare(right),
  );
  const copyPlan = buildDatasetCopyPlan(selectedKeys);

  if (selectedKeys.length === 0) {
    throw new Error(
      'Select at least one key to include in the processed dataset.',
    );
  }

  // Pre-count total demos for accurate progress reporting.
  let overallDemoCount = 0;
  const demoNamesByEntry: string[][] = [];
  for (const entry of entries) {
    const names =
      request.operation === 'cut'
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
        const targetDemoGroup = outputDataGroup.create_group(
          `demo_${outputDemoIndex}`,
          true,
        );

        copyAttributes(sourceDemoGroup, targetDemoGroup);
        const capturedDemoIndex = overallDemoIndex;
        copySelectedDatasetsForDemo(
          sourceDemoGroup,
          targetDemoGroup,
          copyPlan,
          (datasetPath, copiedRows, totalRows) => {
            reportProgress({
              phase: 'copying',
              overallDemoIndex: capturedDemoIndex,
              overallDemoCount,
              currentSourceName: entry.datasetName,
              currentDemoName: sourceDemoName,
              datasetDetail: { path: datasetPath, copiedRows, totalRows },
            });
          },
        );

        totalSamples +=
          optionalInt(getAttributeValue(sourceDemoGroup, 'num_samples')) ?? 0;
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
          {
            id: messageId,
            type: 'chunk',
            data: arrayBuffer,
            index: i,
            total: totalChunks,
          },
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

function listDemoVideoInfo(
  entry: OpenSourceEntry,
  demoName: string,
): DemoVideoInfo[] {
  const demoGroup = getDemoGroup(entry, demoName);
  const videos: DemoVideoInfo[] = [];

  const cameraNames = listCameraNames(demoGroup);
  cameraNames.sort((left, right) => left.localeCompare(right));

  for (const key of cameraNames) {
    const dataset = findVideoDataset(demoGroup, key);
    if (!dataset || !isVideoShape(dataset.shape)) {
      continue;
    }

    const [frameCount, height, width, channels] = dataset.shape;
    videos.push({
      key,
      label: humanizeCameraLabel(key),
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
    copied.set(
      new Uint8Array(value.buffer, value.byteOffset, value.byteLength),
    );
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
      `Demo '${demoName}' does not contain a supported video dataset for camera '${videoKey}' ` +
        `(expected at obs/cameras/${videoKey} or obs/${videoKey}).`,
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
    label: humanizeCameraLabel(videoKey),
    path: dataset.path,
    frameCount,
    height,
    width,
    channels,
    framesBuffer,
  };
}

async function openLocalSource(file: File): Promise<OpenSourceResultPayload> {
  const module = await ensureModule();
  const { FS } = module;
  const mountPoint = `/${uniqueName(`workerfs-${sanitizeFilename(file.name)}`)}`;
  const workerFsPath = `${mountPoint}/${file.name}`;

  FS.mkdir(mountPoint);
  FS.mount(FS.filesystems.WORKERFS, { files: [file] }, mountPoint);

  const h5File = new h5wasm.File(workerFsPath, 'r');
  const demos = listDemos(h5File);
  const articulation = readArticulation(h5File);
  const sourceId = uniqueName('source');
  const datasetName = stripExtension(file.name);

  openSources.set(sourceId, {
    datasetName,
    demos,
    articulation,
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

  return { sourceId, datasetName, demos, articulation };
}

async function openRemoteSource(
  buffer: ArrayBuffer,
  name: string,
): Promise<OpenSourceResultPayload> {
  const module = await ensureModule();
  const { FS } = module;
  const fsName = `/${uniqueName(sanitizeFilename(name))}`;

  FS.writeFile(fsName, new Uint8Array(buffer), { flags: 'w+' });
  const h5File = new h5wasm.File(fsName, 'r');
  const demos = listDemos(h5File);
  const articulation = readArticulation(h5File);
  const sourceId = uniqueName('source');
  const datasetName = stripExtension(name);

  openSources.set(sourceId, {
    datasetName,
    demos,
    articulation,
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

  return { sourceId, datasetName, demos, articulation };
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

async function handleRequest(
  message: PoseTraceWorkerRequest,
): Promise<WorkerSuccessResult> {
  switch (message.type) {
    case 'openLocalSource':
      return { result: await openLocalSource(message.payload.file) };
    case 'openRemoteSource':
      return {
        result: await openRemoteSource(
          message.payload.buffer,
          message.payload.name,
        ),
      };
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
    case 'inspectSourceFeatures': {
      const entry = openSources.get(message.payload.sourceId);
      if (!entry) {
        throw new Error('Pose Trace source is no longer available.');
      }
      return { result: inspectSourceFeatures(entry) };
    }
    case 'loadDatasetComparisonValues': {
      const entry = openSources.get(message.payload.sourceId);
      if (!entry) {
        throw new Error('Pose Trace source is no longer available.');
      }
      return {
        result: loadDatasetComparisonValues(
          entry,
          message.payload.demoName,
          message.payload.keyPaths,
        ),
      };
    }
    case 'loadObjectDistribution':
      return { result: loadObjectDistribution(message.payload) };
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
