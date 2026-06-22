import { describe, expect, it } from 'vitest';

import { humanizeColumnName } from './schema';

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
    expect(humanizeColumnName('keypoint_object_left_lower_z')).not.toContain(
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
