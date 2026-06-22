import { describe, expect, it } from 'vitest';

import { buildFailureAnalysis } from './failureAnalysis';
import type {
  ObjectDistributionCategory,
  ObjectDistributionPoint,
  ObjectDistributionResult,
} from './types';

function makePoint(
  category: ObjectDistributionCategory,
  x: number,
  y: number,
  rx: number,
  ry: number,
): ObjectDistributionPoint {
  return {
    category,
    datasetName: 'ds',
    demoName: `demo_${x}_${y}`,
    x,
    y,
    initialX: x,
    initialY: y,
    initialRx: rx,
    initialRy: ry,
    numSamples: null,
    sourceLeft: '',
    sourceRight: '',
    sourceLeftDetails: [],
    sourceRightDetails: [],
  };
}

function makeResult(
  successPoints: ObjectDistributionPoint[],
  failedPoints: ObjectDistributionPoint[],
  teleopPoints: ObjectDistributionPoint[] = [],
): ObjectDistributionResult {
  return {
    anchor: 'initial_pose',
    successPoints,
    failedPoints,
    teleopPoints,
    teleopDiagnostics: null,
    availableObjects: [],
  };
}

describe('buildFailureAnalysis', () => {
  it('returns null when no generated points have valid reset coordinates', () => {
    const broken: ObjectDistributionPoint = {
      ...makePoint('success', 0, 0, 0, 0),
      initialX: null,
      initialY: null,
      initialRx: null,
      initialRy: null,
    };
    expect(
      buildFailureAnalysis(makeResult([broken], []), { minGeneratedCount: 1 }),
    ).toBeNull();
  });

  it('returns null when both successPoints and failedPoints are empty', () => {
    expect(
      buildFailureAnalysis(makeResult([], []), { minGeneratedCount: 1 }),
    ).toBeNull();
  });

  it('produces position+rotation maps and 9 slices for a small valid input', () => {
    const success = Array.from({ length: 8 }, (_, i) =>
      makePoint('success', i * 0.1, i * 0.1, i * 5, i * 5),
    );
    const failed = Array.from({ length: 8 }, (_, i) =>
      makePoint('failed', i * 0.1 + 0.05, i * 0.1 + 0.05, i * 5 + 1, i * 5 + 1),
    );
    const teleop = Array.from({ length: 4 }, (_, i) =>
      makePoint('teleop', i * 0.2, i * 0.2, i * 7, i * 7),
    );

    const result = buildFailureAnalysis(makeResult(success, failed, teleop), {
      minGeneratedCount: 1,
    });
    expect(result).not.toBeNull();
    if (!result) {
      return;
    }

    expect(result.stats.successGeneratedCount).toBe(8);
    expect(result.stats.failedGeneratedCount).toBe(8);
    expect(result.stats.analyzedGeneratedCount).toBe(16);
    expect(result.stats.teleopCount).toBe(4);

    expect(result.slices).toHaveLength(9);
    for (const slice of result.slices) {
      expect(slice.plane.bins.length).toBeGreaterThan(0);
    }

    expect(result.positionMap.bins.length).toBeGreaterThan(0);
    expect(result.rotationMap.bins.length).toBeGreaterThan(0);
  });

  it('caps recommendations at the documented MAX_RECOMMENDATIONS', () => {
    // Spread points across a wide grid so every cell has enough support.
    const success: ObjectDistributionPoint[] = [];
    const failed: ObjectDistributionPoint[] = [];
    for (let xi = 0; xi < 10; xi += 1) {
      for (let yi = 0; yi < 10; yi += 1) {
        success.push(makePoint('success', xi * 0.1, yi * 0.1, xi * 3, yi * 3));
        failed.push(makePoint('failed', xi * 0.1, yi * 0.1, xi * 3, yi * 3));
        failed.push(makePoint('failed', xi * 0.1, yi * 0.1, xi * 3, yi * 3));
      }
    }

    const result = buildFailureAnalysis(makeResult(success, failed), {
      minGeneratedCount: 1,
    });
    expect(result).not.toBeNull();
    if (!result) {
      return;
    }
    expect(result.recommendations.length).toBeLessThanOrEqual(10);
  });

  it('counts skipped points whose reset coords are null', () => {
    const goodSuccess = makePoint('success', 0.1, 0.1, 0, 0);
    const skippedFailed = {
      ...makePoint('failed', 0, 0, 0, 0),
      initialX: null,
    } as ObjectDistributionPoint;

    const result = buildFailureAnalysis(
      makeResult([goodSuccess], [skippedFailed]),
      { minGeneratedCount: 1 },
    );
    expect(result).not.toBeNull();
    if (!result) {
      return;
    }
    expect(result.stats.skippedGeneratedCount).toBe(1);
    expect(result.stats.analyzedFailedCount).toBe(0);
  });
});
