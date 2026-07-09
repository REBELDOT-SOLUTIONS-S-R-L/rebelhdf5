import { describe, expect, it } from 'vitest';

import {
  build3DData,
  build3DDataForStep,
  build3DLayout,
  buildObjectDistributionData,
  buildObjectDistributionLayout,
  buildCombinedJointChartData,
  buildEmptyLayout,
  buildJointChartData,
  buildJointChartLayout,
  getJointChartSpecs,
} from './plotConfig';
import type {
  ArticulationSegment,
  ObjectDistributionPoint,
  ObjectDistributionResult,
  DemoRow,
} from './types';

function makeRow(step: number, overrides: Partial<DemoRow> = {}): DemoRow {
  return {
    dataset_name: 'ds',
    demo_name: 'demo_0',
    step,
    env_id: 0,
    episode_index: null,
    episode_step: step,
    source_episode_index: null,
    num_samples: 100,
    success: null,
    completed_attempts: null,
    completed_successes: null,
    eef_left_arm_z: 0.5 + step * 0.01,
    eef_right_arm_z: 0.5 + step * 0.01,
    keypoint_object_left_lower_z: 0.4,
    keypoint_object_left_middle_z: 0.4,
    keypoint_object_left_upper_z: 0.4,
    keypoint_object_right_lower_z: 0.4,
    keypoint_object_right_middle_z: 0.4,
    keypoint_object_right_upper_z: 0.4,
    dist_left_arm_to_object_left_middle_m: 0.1,
    dist_left_arm_to_object_left_lower_m: 0.1,
    dist_left_arm_to_object_left_upper_m: 0.1,
    dist_right_arm_to_object_right_middle_m: 0.1,
    dist_right_arm_to_object_right_lower_m: 0.1,
    dist_right_arm_to_object_right_upper_m: 0.1,
    eef_left_arm_x: 0.1,
    eef_left_arm_y: 0.1,
    eef_right_arm_x: -0.1,
    eef_right_arm_y: 0.1,
    ...overrides,
  };
}

describe('buildEmptyLayout', () => {
  it('returns a 2D layout with the title and message annotated', () => {
    const layout = buildEmptyLayout('No data', 'Please pick a file');
    expect(layout.title).toEqual(expect.objectContaining({ text: 'No data' }));
    expect(layout.annotations?.[0]).toMatchObject({
      text: 'Please pick a file',
    });
    expect(layout.height).toBe(520);
    // 2D layout hides plain xaxis, doesn't add a 3D scene.
    expect(layout.xaxis).toMatchObject({ visible: false });
    expect(layout.scene).toBeUndefined();
  });

  it('returns a 3D layout with hidden scene axes when is3d=true', () => {
    const layout = buildEmptyLayout('Empty', 'No file', true);
    expect(layout.scene).toBeDefined();
    expect(layout.height).toBe(760);
    expect(layout.xaxis).toBeUndefined();
  });
});

describe('getJointChartSpecs', () => {
  function segment(
    overrides: Partial<ArticulationSegment> = {},
  ): ArticulationSegment {
    return {
      name: 'left_arm',
      targetStart: 0,
      targetEnd: 4,
      obsStart: 0,
      obsEnd: 4,
      ...overrides,
    };
  }

  it('expands a single segment into one spec per joint (inclusive range)', () => {
    const specs = getJointChartSpecs([segment()]);
    expect(specs).toHaveLength(5);
    expect(specs.map((spec) => spec.jointIndex)).toEqual([0, 1, 2, 3, 4]);
  });

  it('uses the smaller of target and obs lengths when ranges differ', () => {
    const specs = getJointChartSpecs([segment({ targetEnd: 5, obsEnd: 3 })]);
    expect(specs).toHaveLength(4);
  });

  it('builds specs from articulation joint indices with name and index labels', () => {
    const specs = getJointChartSpecs(
      [],
      [
        { articulationName: 'robot', name: 'shoulder_pan', index: 2 },
        { articulationName: 'robot', name: 'elbow', index: 5 },
      ],
    );
    expect(specs).toHaveLength(2);
    expect(specs.map((spec) => spec.label)).toEqual([
      'robot / shoulder_pan [2]',
      'robot / elbow [5]',
    ]);
  });
});

describe('buildJointChartData', () => {
  it('returns no traces for empty rows', () => {
    expect(
      buildJointChartData([], { segmentName: 'left_arm', jointIndex: 0 }),
    ).toEqual([]);
  });

  it('returns one trace per non-empty joint series', () => {
    const rows = Array.from({ length: 3 }, (_, i) =>
      makeRow(i, {
        joint_target_left_arm_0: i * 0.1,
        joint_obs_left_arm_0: i * 0.1 + 0.01,
      }),
    );
    const traces = buildJointChartData(rows, {
      segmentName: 'left_arm',
      jointIndex: 0,
    }) as Array<{ name?: string }>;
    expect(traces).toHaveLength(2);
    expect(traces.map((trace) => trace.name)).toEqual(['target', 'obs']);
  });

  it('combines selected joint target and obs series into one chart', () => {
    const specs = getJointChartSpecs(
      [],
      [
        { articulationName: 'robot', name: 'shoulder_pan', index: 0 },
        { articulationName: 'robot', name: 'elbow', index: 1 },
      ],
    );
    const rows = Array.from({ length: 3 }, (_, i) =>
      makeRow(i, {
        'joint_target_robot::shoulder_pan::0': i,
        'joint_obs_robot::shoulder_pan::0': i + 0.1,
        'joint_target_robot::elbow::1': i + 1,
        'joint_obs_robot::elbow::1': i + 1.1,
      }),
    );
    const traces = buildCombinedJointChartData(rows, specs) as Array<{
      name?: string;
    }>;
    expect(traces.map((trace) => trace.name)).toEqual([
      'robot / shoulder_pan [0] target',
      'robot / shoulder_pan [0] obs',
      'robot / elbow [1] target',
      'robot / elbow [1] obs',
    ]);
  });
});

describe('buildJointChartLayout', () => {
  it('titles the chart with `<segment>: <joint>`', () => {
    const layout = buildJointChartLayout({
      segmentName: 'left_arm',
      jointIndex: 3,
    });
    const title = layout.title as { text?: string } | undefined;
    expect(title?.text).toBe('left_arm: 3');
  });
});

describe('build3DData / build3DDataForStep', () => {
  it('returns [] for empty rows', () => {
    expect(build3DData([])).toEqual([]);
    expect(build3DDataForStep([], 0)).toEqual([]);
  });

  it('builds traces from non-empty rows', () => {
    const rows = Array.from({ length: 4 }, (_, i) => makeRow(i));
    const traces = build3DData(rows);
    expect(traces.length).toBeGreaterThan(0);
  });
});

describe('build3DLayout', () => {
  it('uses the default camera when none provided', () => {
    const layout = build3DLayout([makeRow(0)]) as {
      scene?: { camera?: { eye?: { y?: number } } };
    };
    expect(layout.scene?.camera?.eye?.y).toBeDefined();
  });

  it('respects a caller-provided camera', () => {
    const camera = { eye: { x: 1, y: 2, z: 3 } };
    const layout = build3DLayout([makeRow(0)], camera) as {
      scene?: { camera?: typeof camera };
    };
    expect(layout.scene?.camera).toEqual(camera);
  });
});

describe('buildObjectDistribution helpers', () => {
  function emptyResult(): ObjectDistributionResult {
    return {
      anchor: 'initial_pose',
      successPoints: [],
      failedPoints: [],
      teleopPoints: [],
      teleopDiagnostics: null,
      availableObjects: [],
    };
  }

  function makePoint(
    category: 'success' | 'failed' | 'teleop',
    x = 0,
    y = 0,
  ): ObjectDistributionPoint {
    return {
      category,
      datasetName: 'ds',
      demoName: 'demo_0',
      x,
      y,
      initialX: x,
      initialY: y,
      initialRx: 0,
      initialRy: 0,
      numSamples: null,
      sourceLeft: '',
      sourceRight: '',
      sourceLeftDetails: [],
      sourceRightDetails: [],
    };
  }

  it('returns no traces for a null result', () => {
    expect(buildObjectDistributionData(null, null)).toEqual([]);
  });

  it('returns category traces when points are present, no selection', () => {
    const result = emptyResult();
    result.successPoints.push(makePoint('success', 0.1, 0.1));
    result.failedPoints.push(makePoint('failed', 0.2, 0.2));
    const traces = buildObjectDistributionData(result, null);
    expect(traces.length).toBeGreaterThanOrEqual(2);
  });

  it('appends selected-episode trace when a non-teleop point is selected', () => {
    const result = emptyResult();
    const selected = makePoint('success', 0.1, 0.1);
    result.successPoints.push(selected);
    const tracesWithoutSelection = buildObjectDistributionData(result, null);
    const tracesWithSelection = buildObjectDistributionData(result, selected);
    expect(tracesWithSelection.length).toBeGreaterThan(
      tracesWithoutSelection.length,
    );
  });

  it('does not append a selected-episode trace for teleop selection', () => {
    const result = emptyResult();
    const teleop = makePoint('teleop', 0.1, 0.1);
    result.teleopPoints.push(teleop);
    const traces = buildObjectDistributionData(result, teleop);
    expect(traces).toHaveLength(1);
  });

  it('returns a layout regardless of result presence', () => {
    expect(buildObjectDistributionLayout(null, 'initial_pose').height).toBe(
      820,
    );
    expect(
      buildObjectDistributionLayout(emptyResult(), 'initial_pose').height,
    ).toBe(820);
  });
});
