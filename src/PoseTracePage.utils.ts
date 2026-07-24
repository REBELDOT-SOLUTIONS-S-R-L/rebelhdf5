import {
  type JointChartSpec,
  type PlotSceneCamera,
} from './pose-trace/plotConfig';
import { type DemoInfo } from './pose-trace/types';

/** Build a human-readable label for a demo option in the dropdown. */
export function formatDemoOption(demo: DemoInfo): string {
  const parts = [demo.name];
  if (demo.num_samples !== null) {
    parts.push(`samples=${demo.num_samples}`);
  }
  if (demo.success !== null) {
    parts.push(`success=${demo.success ? 1 : 0}`);
  }
  if (demo.source_episode_index !== null) {
    parts.push(`source=${demo.source_episode_index}`);
  }
  return parts.join(' | ');
}

/** Deep-clone a Plotly 3D scene camera so stored state is not mutated. */
export function clonePlotSceneCamera(
  camera: PlotSceneCamera | null | undefined,
): PlotSceneCamera | null {
  if (!camera) {
    return null;
  }

  return {
    center: camera.center ? { ...camera.center } : undefined,
    eye: camera.eye ? { ...camera.eye } : undefined,
    up: camera.up ? { ...camera.up } : undefined,
    projection: camera.projection ? { ...camera.projection } : undefined,
  };
}

/** Stable identity key for a joint chart spec (falls back to segment+index). */
export function jointSpecKey(spec: JointChartSpec): string {
  return spec.id ?? `${spec.segmentName}_${spec.jointIndex}`;
}

/** Display label for a joint chart spec (falls back to segment: index). */
export function jointSpecLabel(spec: JointChartSpec): string {
  return spec.label ?? `${spec.segmentName}: ${spec.jointIndex}`;
}
