import { describe, expect, it } from 'vitest';

import {
  analysisTitle,
  classifyDataset,
  datasetBaseName,
  formatPercent,
  formatRange,
  formatSupport,
  hashRevisionKey,
  type PlotClickEvent,
  resolveClickedPoint,
  selectDatasetGroup,
} from './ObjectDistributionPage.utils';
import {
  type ObjectDistributionPoint,
  type ObjectDistributionResult,
} from './pose-trace/types';
import { FileService, type H5File } from './stores';

function remoteFile(name: string, url = name): H5File {
  return {
    url,
    name,
    service: FileService.Url,
    resolvedUrl: `https://example.com/${name}`,
  };
}

describe('classifyDataset', () => {
  it('classifies failed generated datasets (failed wins over generated)', () => {
    expect(classifyDataset(remoteFile('cube_generated_run_failed.hdf5'))).toBe(
      'failed',
    );
  });

  it('classifies successful generated datasets', () => {
    expect(classifyDataset(remoteFile('cube_generated_run.hdf5'))).toBe(
      'success',
    );
  });

  it('classifies annotated/teleop datasets', () => {
    expect(classifyDataset(remoteFile('cube_annotated_run.hdf5'))).toBe(
      'teleop',
    );
    expect(classifyDataset(remoteFile('cube_teleop_demo.hdf5'))).toBe('teleop');
  });

  it('returns null for names matching no convention', () => {
    expect(classifyDataset(remoteFile('random_dataset.hdf5'))).toBeNull();
  });

  it('is case-insensitive', () => {
    expect(classifyDataset(remoteFile('CUBE_GENERATED_FAILED.HDF5'))).toBe(
      'failed',
    );
  });
});

describe('datasetBaseName', () => {
  it('extracts the shared prefix before the role marker', () => {
    expect(datasetBaseName(remoteFile('cube_stack_generated_01.hdf5'))).toBe(
      'cube_stack',
    );
    expect(datasetBaseName(remoteFile('cube_stack_annotated_01.hdf5'))).toBe(
      'cube_stack',
    );
  });

  it('falls back to the extension-stripped stem when no marker is present', () => {
    expect(datasetBaseName(remoteFile('plain_dataset.hdf5'))).toBe(
      'plain_dataset',
    );
  });
});

describe('selectDatasetGroup', () => {
  const success = remoteFile('cube_generated_a.hdf5', 'u-success');
  const failed = remoteFile('cube_generated_a_failed.hdf5', 'u-failed');
  const teleop = remoteFile('cube_annotated_a.hdf5', 'u-teleop');

  it('assigns each role from a complete pack', () => {
    const selection = selectDatasetGroup([success, failed, teleop], null);
    expect(selection).toEqual({
      successUrl: 'u-success',
      failedUrl: 'u-failed',
      teleopUrl: 'u-teleop',
    });
  });

  it('prefers the pack containing the preferred url', () => {
    const otherSuccess = remoteFile('block_generated_b.hdf5', 'u-other');
    const selection = selectDatasetGroup(
      [success, failed, teleop, otherSuccess],
      'u-other',
    );
    // Preferred belongs to the `block` pack, which only has a success file.
    expect(selection).toEqual({
      successUrl: 'u-other',
      failedUrl: null,
      teleopUrl: null,
    });
  });

  it('picks the pack with the most distinct roles when none is preferred', () => {
    const lonelySuccess = remoteFile('block_generated_b.hdf5', 'u-lonely');
    const selection = selectDatasetGroup(
      [lonelySuccess, success, failed, teleop],
      null,
    );
    // The `cube` pack has 3 roles vs `block`'s 1, so cube wins.
    expect(selection.successUrl).toBe('u-success');
    expect(selection.failedUrl).toBe('u-failed');
    expect(selection.teleopUrl).toBe('u-teleop');
  });

  it('leaves roles null when no file matches', () => {
    const selection = selectDatasetGroup([remoteFile('mystery.hdf5')], null);
    expect(selection).toEqual({
      successUrl: null,
      failedUrl: null,
      teleopUrl: null,
    });
  });
});

describe('formatters', () => {
  it('formatPercent multiplies by 100 with one decimal', () => {
    expect(formatPercent(0.1234)).toBe('12.3%');
    expect(formatPercent(1)).toBe('100.0%');
  });

  it('formatSupport uses two decimals', () => {
    expect(formatSupport(3)).toBe('3.00');
    expect(formatSupport(1.239)).toBe('1.24');
  });

  it('formatRange renders "start to end unit"', () => {
    expect(formatRange(0.1, 0.5, 'm')).toBe('0.10 to 0.50 m');
  });

  it('analysisTitle maps tabs to titles', () => {
    expect(analysisTitle('position')).toBe('Position Failure Map');
    expect(analysisTitle('rotation')).toBe('Rotation Failure Map');
    expect(analysisTitle('slices')).toBe('Spatial Failure Slices');
    expect(analysisTitle('scatter')).toBe('Spatial Failure Slices');
  });
});

describe('hashRevisionKey', () => {
  it('is deterministic for equal inputs', () => {
    expect(hashRevisionKey(['a', 1, true])).toBe(
      hashRevisionKey(['a', 1, true]),
    );
  });

  it('differs for different inputs', () => {
    expect(hashRevisionKey(['a', 1])).not.toBe(hashRevisionKey(['a', 2]));
  });

  it('returns 0 for an empty parts list', () => {
    expect(hashRevisionKey([])).toBe(0);
  });
});

describe('resolveClickedPoint', () => {
  const successPoint = {
    datasetName: 's',
  } as unknown as ObjectDistributionPoint;
  const failedPoint = {
    datasetName: 'f',
  } as unknown as ObjectDistributionPoint;
  const result = {
    successPoints: [successPoint],
    failedPoints: [failedPoint],
  } as unknown as ObjectDistributionResult;

  it('returns the success point for a Success series click', () => {
    const event: PlotClickEvent = {
      points: [{ pointIndex: 0, data: { name: 'Success' } }],
    };
    expect(resolveClickedPoint(event, result)).toBe(successPoint);
  });

  it('returns the failed point for a Failed series click', () => {
    const event: PlotClickEvent = {
      points: [{ pointIndex: 0, data: { name: 'Failed' } }],
    };
    expect(resolveClickedPoint(event, result)).toBe(failedPoint);
  });

  it('returns null when result is null', () => {
    const event: PlotClickEvent = {
      points: [{ pointIndex: 0, data: { name: 'Success' } }],
    };
    expect(resolveClickedPoint(event, null)).toBeNull();
  });

  it('returns null with no points or negative index', () => {
    expect(resolveClickedPoint({}, result)).toBeNull();
    expect(
      resolveClickedPoint(
        { points: [{ pointIndex: -1, data: { name: 'Success' } }] },
        result,
      ),
    ).toBeNull();
  });

  it('returns null for an out-of-range index', () => {
    const event: PlotClickEvent = {
      points: [{ pointIndex: 9, data: { name: 'Success' } }],
    };
    expect(resolveClickedPoint(event, result)).toBeNull();
  });

  it('returns null for an unknown series name', () => {
    const event: PlotClickEvent = {
      points: [{ pointIndex: 0, data: { name: 'Teleop' } }],
    };
    expect(resolveClickedPoint(event, result)).toBeNull();
  });
});
