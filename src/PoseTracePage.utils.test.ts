import { describe, expect, it } from 'vitest';

import {
  type JointChartSpec,
  type PlotSceneCamera,
} from './pose-trace/plotConfig';
import { type DemoInfo } from './pose-trace/types';
import {
  clonePlotSceneCamera,
  formatDemoOption,
  jointSpecKey,
  jointSpecLabel,
} from './PoseTracePage.utils';

function demo(overrides: Partial<DemoInfo>): DemoInfo {
  return {
    name: 'demo_0',
    episode_index: null,
    num_samples: null,
    success: null,
    source_episode_index: null,
    ...overrides,
  };
}

describe('formatDemoOption', () => {
  it('shows just the name when no metadata is present', () => {
    expect(formatDemoOption(demo({ name: 'demo_2' }))).toBe('demo_2');
  });

  it('includes samples, success and source metadata', () => {
    expect(
      formatDemoOption(
        demo({
          name: 'demo_1',
          num_samples: 10,
          success: false,
          source_episode_index: 3,
        }),
      ),
    ).toBe('demo_1 | samples=10 | success=0 | source=3');
  });
});

describe('clonePlotSceneCamera', () => {
  it('returns null for null/undefined input', () => {
    expect(clonePlotSceneCamera(null)).toBeNull();
    expect(clonePlotSceneCamera(undefined)).toBeNull();
  });

  it('deep-clones present sub-objects and leaves absent ones undefined', () => {
    const camera: PlotSceneCamera = {
      eye: { x: 1, y: 2, z: 3 },
      projection: { type: 'perspective' },
    };
    const clone = clonePlotSceneCamera(camera);
    expect(clone).toEqual({
      center: undefined,
      eye: { x: 1, y: 2, z: 3 },
      up: undefined,
      projection: { type: 'perspective' },
    });
    // Nested objects are copies, not references.
    expect(clone?.eye).not.toBe(camera.eye);
    expect(clone?.projection).not.toBe(camera.projection);
  });
});

describe('jointSpecKey / jointSpecLabel', () => {
  it('prefers the explicit id / label when present', () => {
    const spec: JointChartSpec = {
      id: 'custom-id',
      label: 'Custom Label',
      segmentName: 'arm',
      jointIndex: 2,
    };
    expect(jointSpecKey(spec)).toBe('custom-id');
    expect(jointSpecLabel(spec)).toBe('Custom Label');
  });

  it('falls back to segment name and joint index', () => {
    const spec: JointChartSpec = { segmentName: 'arm', jointIndex: 2 };
    expect(jointSpecKey(spec)).toBe('arm_2');
    expect(jointSpecLabel(spec)).toBe('arm: 2');
  });
});
