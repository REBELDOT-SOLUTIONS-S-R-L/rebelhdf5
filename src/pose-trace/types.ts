export interface DemoInfo {
  name: string;
  episode_index: number | null;
  num_samples: number | null;
  success: boolean | null;
  source_episode_index: number | null;
}

export const DEMO_VIDEO_KEYS = ['left_wrist', 'right_wrist', 'top'] as const;
export type DemoVideoKey = (typeof DEMO_VIDEO_KEYS)[number];

export interface DemoVideoInfo {
  key: DemoVideoKey;
  path: string;
  frameCount: number;
  height: number;
  width: number;
  channels: number;
}

export interface DemoVideoFrames extends DemoVideoInfo {
  frames: Uint8Array;
}

export type DatasetProcessingOperation = 'cut' | 'merge' | 'append';

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

export interface DatasetProcessingResult {
  fileName: string;
  demoCount: number;
  selectedKeyCount: number;
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

export interface PoseTraceSource {
  sourceId: string;
  datasetName: string;
  demos: DemoInfo[];
  cleanup: () => void;
}
