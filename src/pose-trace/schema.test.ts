import { describe, expect, it } from 'vitest';

import {
  humanizeColumnName,
  TRACE_3D_SPECS,
  TRACE_EEF_KEYPOINT_COLUMNS,
  TRACE_EEF_KEYPOINT_GROUPS,
  TRACE_EEF_NAMES,
  TRACE_KEYPOINT_NAMES,
  TRACE_SUCCESS_DISTANCE_SPECS,
  TRACE_TERM_COLUMNS,
  TRACE_Z_COLUMNS,
} from './schema';

describe('humanizeColumnName', () => {
  it('strips the dist_ prefix', () => {
    expect(humanizeColumnName('dist_left_arm_z')).not.toContain('dist_');
  });

  it('strips the threshold_ prefix', () => {
    expect(humanizeColumnName('threshold_left_arm_z')).not.toContain(
      'threshold_',
    );
  });

  it('strips the keypoint_ prefix', () => {
    expect(humanizeColumnName('keypoint_garment_left_lower_z')).not.toContain(
      'keypoint_',
    );
  });

  it('strips the eef_ prefix', () => {
    expect(humanizeColumnName('eef_left_arm_z')).toBe('left arm z');
  });

  it('replaces remaining underscores with spaces', () => {
    expect(humanizeColumnName('a_b_c')).toBe('a b c');
  });
});

describe('schema constants', () => {
  it('lists exactly the two end-effector arms', () => {
    expect(TRACE_EEF_NAMES).toEqual(['left_arm', 'right_arm']);
  });

  it('lists six garment keypoints', () => {
    expect(TRACE_KEYPOINT_NAMES).toHaveLength(6);
  });

  it('groups three garment keypoints under each arm', () => {
    expect(TRACE_EEF_KEYPOINT_GROUPS.left_arm).toHaveLength(3);
    expect(TRACE_EEF_KEYPOINT_GROUPS.right_arm).toHaveLength(3);
  });

  it('keeps every keypoint group entry inside the global keypoint list', () => {
    const all = new Set<string>(TRACE_KEYPOINT_NAMES);
    for (const arm of TRACE_EEF_NAMES) {
      for (const kp of TRACE_EEF_KEYPOINT_GROUPS[arm]) {
        expect(all.has(kp)).toBe(true);
      }
    }
  });

  it('builds a (dist, threshold) pair per success-distance spec', () => {
    expect(TRACE_TERM_COLUMNS).toHaveLength(TRACE_SUCCESS_DISTANCE_SPECS.length);
    for (const [dist, threshold] of TRACE_TERM_COLUMNS) {
      expect(dist.startsWith('dist_')).toBe(true);
      expect(threshold.startsWith('threshold_')).toBe(true);
    }
  });

  it('lists Z columns for both eef arms and all six keypoints', () => {
    expect(TRACE_Z_COLUMNS).toHaveLength(2 + 6);
  });

  it('keeps EEF keypoint distance columns aligned with the keypoint list', () => {
    expect(TRACE_EEF_KEYPOINT_COLUMNS).toHaveLength(
      TRACE_EEF_NAMES.length * 3,
    );
  });

  it('gives every 3D trace spec a non-empty label, color, and prefix', () => {
    for (const spec of TRACE_3D_SPECS) {
      expect(spec.label.length).toBeGreaterThan(0);
      expect(spec.color).toMatch(/^#[\dA-Fa-f]{6}$/u);
      expect(spec.prefix.length).toBeGreaterThan(0);
      expect(spec.markerSize).toBeGreaterThan(0);
    }
  });
});
