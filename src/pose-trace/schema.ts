export interface Trace3DSpec {
  label: string;
  prefix: string;
  color: string;
  dash: string;
  markerSize: number;
}

export function humanizeColumnName(column: string): string {
  let label = column;
  for (const prefix of [
    'dist_',
    'threshold_',
    'target_eef_',
    'ik_input_eef_',
    'eef_post_step_',
    'keypoint_',
    'object_',
    'eef_',
  ]) {
    label = label.replaceAll(prefix, '');
  }
  label = label.replaceAll('_m', '');
  label = label.replaceAll('_', ' ');
  return label;
}
