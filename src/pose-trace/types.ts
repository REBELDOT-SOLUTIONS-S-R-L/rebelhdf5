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

export const CLOTH_DISTRIBUTION_DIRECT_ANCHORS = [
  'garment_left_lower',
  'garment_left_middle',
  'garment_left_upper',
  'garment_right_lower',
  'garment_right_middle',
  'garment_right_upper',
] as const;

export const CLOTH_DISTRIBUTION_DERIVED_ANCHORS = [
  'garment_center',
  'garment_lower_center',
  'garment_upper_center',
] as const;

export const CLOTH_DISTRIBUTION_ANCHORS = [
  'initial_pose',
  ...CLOTH_DISTRIBUTION_DIRECT_ANCHORS,
  ...CLOTH_DISTRIBUTION_DERIVED_ANCHORS,
] as const;

export type ClothDistributionAnchor = (typeof CLOTH_DISTRIBUTION_ANCHORS)[number];
export const DEFAULT_CLOTH_DISTRIBUTION_ANCHOR: ClothDistributionAnchor = 'garment_center';

export type ClothDistributionCategory = 'success' | 'failed' | 'teleop';

export interface ClothDistributionSourceDetail {
  teleopId: string;
  datasetName: string;
  demoName: string;
  x: number;
  y: number;
  slots: number[];
  textLabel: string;
  hoverLabel: string;
}

export interface ClothDistributionPoint {
  category: ClothDistributionCategory;
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
  sourceLeftDetails: ClothDistributionSourceDetail[];
  sourceRightDetails: ClothDistributionSourceDetail[];
}

export interface ClothDistributionRequest {
  successSourceId: string | null;
  failedSourceId: string | null;
  teleopSourceId: string | null;
  anchor: ClothDistributionAnchor;
  includeRandomSelections: boolean;
}

export interface ClothDistributionSourceDiagnostics {
  totalDemos: number;
  includedDemos: number;
  missingAnchorCount: number;
  missingObjectPositionsCount: number;
}

export interface ClothDistributionResult {
  anchor: ClothDistributionAnchor;
  successPoints: ClothDistributionPoint[];
  failedPoints: ClothDistributionPoint[];
  teleopPoints: ClothDistributionPoint[];
  teleopDiagnostics: ClothDistributionSourceDiagnostics | null;
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
  phase: 'copying' | 'flushing' | 'streaming' | 'converting' | 'encoding' | 'stats' | 'metadata';
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
  gripperStart: number | null;
  gripperEnd: number | null;
}

export interface ParsedArticulation {
  name: string;
  jointNumber: number | null;
  segmentation: ArticulationSegment[];
  endEffectors: ArticulationEndEffector[];
}

export interface PoseTraceSource {
  sourceId: string;
  datasetName: string;
  demos: DemoInfo[];
  articulation: ParsedArticulation | null;
  cleanup: () => void;
}
