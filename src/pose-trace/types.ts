export interface DemoInfo {
  name: string;
  episode_index: number | null;
  num_samples: number | null;
  success: boolean | null;
  source_episode_index: number | null;
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
