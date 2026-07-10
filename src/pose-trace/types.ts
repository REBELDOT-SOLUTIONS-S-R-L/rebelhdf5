export interface DemoInfo {
  name: string;
  episode_index: number | null;
  num_samples: number | null;
  success: boolean | null;
  source_episode_index: number | null;
}

// Camera names are no longer fixed — they are discovered from `obs/cameras/`
// (new standard schema) with legacy fallback to direct datasets under `obs/`.
export type DemoVideoKey = string;

export const OBJECT_DISTRIBUTION_ANCHORS = ['initial_pose'] as const;

export type ObjectDistributionAnchor =
  (typeof OBJECT_DISTRIBUTION_ANCHORS)[number];
export const DEFAULT_OBJECT_DISTRIBUTION_ANCHOR: ObjectDistributionAnchor =
  'initial_pose';

export type ObjectDistributionCategory = 'success' | 'failed' | 'teleop';

export interface ObjectDistributionSourceDetail {
  teleopId: string;
  datasetName: string;
  demoName: string;
  x: number;
  y: number;
  slots: number[];
  textLabel: string;
  hoverLabel: string;
}

export interface ObjectDistributionPoint {
  category: ObjectDistributionCategory;
  datasetName: string;
  demoName: string;
  x: number;
  y: number;
  initialX: number | null;
  initialY: number | null;
  initialRx: number | null;
  initialRy: number | null;
  numSamples: number | null;
  sourceLeft: string;
  sourceRight: string;
  sourceLeftDetails: ObjectDistributionSourceDetail[];
  sourceRightDetails: ObjectDistributionSourceDetail[];
}

export interface ObjectDistributionRequest {
  successSourceId: string | null;
  failedSourceId: string | null;
  teleopSourceId: string | null;
  anchor: ObjectDistributionAnchor;
  /**
   * Name of the rigid object whose `initial_pose` drives the scatter. When
   * `null`, the first object found under `initial_state/rigid_objects` is used
   * (legacy behavior).
   */
  objectName: string | null;
}

export interface ObjectDistributionSourceDiagnostics {
  totalDemos: number;
  includedDemos: number;
  missingAnchorCount: number;
}

export interface ObjectDistributionResult {
  anchor: ObjectDistributionAnchor;
  successPoints: ObjectDistributionPoint[];
  failedPoints: ObjectDistributionPoint[];
  teleopPoints: ObjectDistributionPoint[];
  teleopDiagnostics: ObjectDistributionSourceDiagnostics | null;
  /**
   * Union of rigid-object names discovered across the selected datasets, in
   * the order they appear under `initial_state/rigid_objects`. Used to populate
   * the object selector on the distribution page.
   */
  availableObjects: string[];
}

export interface DemoVideoInfo {
  key: DemoVideoKey;
  label: string;
  path: string;
  frameCount: number;
  height: number;
  width: number;
  channels: number;
}

export interface DemoVideoFrames extends DemoVideoInfo {
  frames: Uint8Array;
}

export type DatasetProcessingOperation = 'cut' | 'merge' | 'append' | 'lerobot';

export interface DatasetProcessingKeyInfo {
  path: string;
  availableInDemoCount: number;
}

export interface DatasetProcessingSourceInfo {
  keyPaths: DatasetProcessingKeyInfo[];
}

export interface SourceFeatureCapabilities {
  demoCount: number;
  keyCount: number;
  poseTraceDemoCount: number;
  videoDemoCount: number;
  videoCount: number;
  objectDistributionDemoCount: number;
}

export interface DatasetComparisonValue {
  keyPath: string;
  status: 'ok' | 'missing' | 'error';
  shape: number[] | null;
  dtype: string | null;
  selection: string | null;
  value: unknown;
  error?: string;
}

export interface DatasetComparisonValuesResult {
  demoName: string;
  values: DatasetComparisonValue[];
}

export interface DatasetProcessingCutRange {
  startDemoName: string;
  endDemoName: string;
}

export interface DatasetProcessingRequest {
  operation: DatasetProcessingOperation;
  orderedSourceIds: string[];
  selectedKeys: string[];
  fileName: string;
  cutRange?: DatasetProcessingCutRange;
}

export interface DatasetProcessingProgress {
  phase:
    | 'copying'
    | 'flushing'
    | 'streaming'
    | 'converting'
    | 'encoding'
    | 'stats'
    | 'metadata';
  overallDemoIndex: number;
  overallDemoCount: number;
  currentSourceName: string;
  currentDemoName: string;
  /** Present when a large dataset is being copied in chunks within a demo. */
  datasetDetail?: {
    path: string;
    copiedRows: number;
    totalRows: number;
  };
}

export interface DatasetProcessingResultMeta {
  fileName: string;
  demoCount: number;
  selectedKeyCount: number;
  outputPath?: string;
  outputType?: 'file' | 'directory';
  skippedDemoCount?: number;
  totalFrames?: number;
  taskCount?: number;
}

/** @deprecated Use DatasetProcessingResultMeta + chunk streaming instead. */
export interface DatasetProcessingResult extends DatasetProcessingResultMeta {
  fileBuffer: ArrayBuffer;
}

export interface DemoRow {
  dataset_name: string;
  demo_name: string;
  step: number;
  env_id: number;
  episode_index: number | null;
  episode_step: number;
  source_episode_index: number | null;
  num_samples: number;
  success: number | null;
  completed_attempts: number | null;
  completed_successes: number | null;
  [key: string]: string | number | null;
}

export interface ArticulationSegment {
  name: string;
  targetStart: number;
  targetEnd: number;
  obsStart: number;
  obsEnd: number;
}

export interface ArticulationEndEffector {
  name: string;
  poseStart: number;
  poseEnd: number;
  poseOrder: string[];
  gripperStart: number | null;
  gripperEnd: number | null;
}

export interface ArticulationJoint {
  articulationName: string;
  name: string;
  index: number;
}

export interface ParsedArticulation {
  name: string;
  jointNumber: number | null;
  segmentation: ArticulationSegment[];
  endEffectors: ArticulationEndEffector[];
  joints: ArticulationJoint[];
}

export interface PoseTraceSource {
  sourceId: string;
  datasetName: string;
  demos: DemoInfo[];
  articulation: ParsedArticulation | null;
  cleanup: () => void;
}
