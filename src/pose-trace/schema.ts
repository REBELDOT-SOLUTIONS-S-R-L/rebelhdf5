export const TRACE_EEF_NAMES = ['left_arm', 'right_arm'] as const;

export const TRACE_KEYPOINT_NAMES = [
  'garment_left_upper',
  'garment_right_upper',
  'garment_left_middle',
  'garment_right_middle',
  'garment_left_lower',
  'garment_right_lower',
] as const;

export const TRACE_EEF_KEYPOINT_GROUPS: Record<
  (typeof TRACE_EEF_NAMES)[number],
  readonly string[]
> = {
  left_arm: ['garment_left_upper', 'garment_left_middle', 'garment_left_lower'],
  right_arm: ['garment_right_upper', 'garment_right_middle', 'garment_right_lower'],
};

export const TRACE_SUCCESS_DISTANCE_SPECS = [
  ['left_middle_to_lower', 'garment_left_middle', 'garment_left_lower', 0.05],
  ['right_middle_to_lower', 'garment_right_middle', 'garment_right_lower', 0.05],
  ['left_lower_to_upper', 'garment_left_lower', 'garment_left_upper', 0.05],
  ['right_lower_to_upper', 'garment_right_lower', 'garment_right_upper', 0.05],
] as const;

export const TRACE_EEF_KEYPOINT_COLUMNS = [
  'dist_left_arm_to_garment_left_middle_m',
  'dist_left_arm_to_garment_left_lower_m',
  'dist_left_arm_to_garment_left_upper_m',
  'dist_right_arm_to_garment_right_middle_m',
  'dist_right_arm_to_garment_right_lower_m',
  'dist_right_arm_to_garment_right_upper_m',
] as const;

export const TRACE_TERM_COLUMNS = TRACE_SUCCESS_DISTANCE_SPECS.map(
  ([metricName]) => [`dist_${metricName}_m`, `threshold_${metricName}_m`] as const,
);

export const TRACE_Z_COLUMNS = [
  'eef_left_arm_z',
  'eef_right_arm_z',
  'keypoint_garment_left_middle_z',
  'keypoint_garment_left_lower_z',
  'keypoint_garment_left_upper_z',
  'keypoint_garment_right_middle_z',
  'keypoint_garment_right_lower_z',
  'keypoint_garment_right_upper_z',
] as const;

export interface Trace3DSpec {
  label: string;
  prefix: string;
  color: string;
  dash: string;
  markerSize: number;
}

export const TRACE_3D_SPECS: Trace3DSpec[] = [
  { label: 'left eef', prefix: 'eef_left_arm', color: '#1f77b4', dash: 'solid', markerSize: 6 },
  { label: 'right eef', prefix: 'eef_right_arm', color: '#d62728', dash: 'solid', markerSize: 6 },
  { label: 'left upper', prefix: 'keypoint_garment_left_upper', color: '#9467bd', dash: 'dash', markerSize: 4 },
  { label: 'left middle', prefix: 'keypoint_garment_left_middle', color: '#17becf', dash: 'dash', markerSize: 4 },
  { label: 'left lower', prefix: 'keypoint_garment_left_lower', color: '#2ca02c', dash: 'dash', markerSize: 4 },
  { label: 'right upper', prefix: 'keypoint_garment_right_upper', color: '#e377c2', dash: 'dash', markerSize: 4 },
  { label: 'right middle', prefix: 'keypoint_garment_right_middle', color: '#ff7f0e', dash: 'dash', markerSize: 4 },
  { label: 'right lower', prefix: 'keypoint_garment_right_lower', color: '#8c564b', dash: 'dash', markerSize: 4 },
];

export function humanizeColumnName(column: string): string {
  let label = column;
  for (const prefix of ['dist_', 'threshold_', 'keypoint_', 'eef_']) {
    label = label.replaceAll(prefix, '');
  }
  label = label.replaceAll('_m', '');
  label = label.replaceAll('_', ' ');
  return label;
}
