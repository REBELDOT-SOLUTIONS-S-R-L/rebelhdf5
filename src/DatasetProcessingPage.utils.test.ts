import { describe, expect, it } from 'vitest';

import {
  buildBackendKeyInfos,
  buildDefaultOutputName,
  buildKeyTree,
  parseTaskRulesJson,
  stripExtension,
  sumKeyInfos,
} from './DatasetProcessingPage.utils';
import { type DatasetProcessingKeyInfo } from './pose-trace/types';
import { type PythonScanResult } from './python-backend';

function fileInfo(
  overrides: Partial<PythonScanResult['files'][number]>,
): PythonScanResult['files'][number] {
  return {
    name: 'f.h5',
    path: '/data/f.h5',
    demoCount: 4,
    demoNames: [],
    keys: [],
    ...overrides,
  };
}

describe('stripExtension', () => {
  it('removes .h5 and .hdf5 case-insensitively', () => {
    expect(stripExtension('dataset.h5')).toBe('dataset');
    expect(stripExtension('dataset.hdf5')).toBe('dataset');
    expect(stripExtension('DATASET.HDF5')).toBe('DATASET');
  });

  it('only strips a trailing extension', () => {
    expect(stripExtension('a.h5.backup')).toBe('a.h5.backup');
    expect(stripExtension('folder.h5/file')).toBe('folder.h5/file');
  });

  it('leaves names without a known extension untouched', () => {
    expect(stripExtension('dataset')).toBe('dataset');
    expect(stripExtension('dataset.txt')).toBe('dataset.txt');
  });
});

describe('parseTaskRulesJson', () => {
  it('returns undefined for blank / whitespace input', () => {
    expect(parseTaskRulesJson('')).toBeUndefined();
    expect(parseTaskRulesJson('   \n\t ')).toBeUndefined();
  });

  it('parses a JSON array of objects', () => {
    const rules = parseTaskRulesJson('[{"match": "a"}, {"match": "b"}]');
    expect(rules).toEqual([{ match: 'a' }, { match: 'b' }]);
  });

  it('throws when the top level is not an array', () => {
    expect(() => parseTaskRulesJson('{"match": "a"}')).toThrow(
      /must be a JSON array of objects/u,
    );
  });

  it('throws when an entry is not a plain object', () => {
    expect(() => parseTaskRulesJson('[{"ok": 1}, 5]')).toThrow(
      /must be a JSON array of objects/u,
    );
    expect(() => parseTaskRulesJson('[[1, 2]]')).toThrow(
      /must be a JSON array of objects/u,
    );
  });

  it('throws on invalid JSON', () => {
    expect(() => parseTaskRulesJson('[not json]')).toThrow(/JSON|Unexpected/u);
  });
});

describe('buildDefaultOutputName', () => {
  it('defaults a single-source lerobot output to v3 without an extension', () => {
    expect(buildDefaultOutputName('lerobot', [{ name: 'run.hdf5' }], [])).toBe(
      'run-lerobot-v3',
    );
  });

  it('names a multi-source lerobot output by count', () => {
    expect(
      buildDefaultOutputName(
        'lerobot',
        [{ name: 'a.h5' }, { name: 'b.h5' }],
        [],
      ),
    ).toBe('lerobot-v3-2-datasets');
  });

  it('uses the legacy suffix when v2.1 is selected', () => {
    expect(
      buildDefaultOutputName('lerobot', [{ name: 'run.hdf5' }], [], 'v2.1'),
    ).toBe('run-lerobot-v21');
  });

  it('names a cut output using first/last demo and strips the extension', () => {
    expect(
      buildDefaultOutputName(
        'cut',
        [{ name: 'run.hdf5' }],
        ['demo_2', 'demo_3', 'demo_5'],
      ),
    ).toBe('run-cut-demo_2-demo_5.hdf5');
  });

  it('falls back to start/end placeholders when no demos are given', () => {
    expect(buildDefaultOutputName('cut', [{ name: 'run.h5' }], [])).toBe(
      'run-cut-start-end.hdf5',
    );
  });

  it('names an append output and pluralizes the dataset count', () => {
    expect(
      buildDefaultOutputName(
        'append',
        [{ name: 'base.h5' }, { name: 'x.h5' }, { name: 'y.h5' }],
        [],
      ),
    ).toBe('base-append-2-datasets.hdf5');
    expect(
      buildDefaultOutputName(
        'append',
        [{ name: 'base.h5' }, { name: 'x.h5' }],
        [],
      ),
    ).toBe('base-append-1-dataset.hdf5');
  });

  it('names a merge output by source count', () => {
    expect(
      buildDefaultOutputName('merge', [{ name: 'a.h5' }, { name: 'b.h5' }], []),
    ).toBe('merged-2-datasets.hdf5');
  });
});

function info(
  path: string,
  availableInDemoCount = 1,
): DatasetProcessingKeyInfo {
  return { path, availableInDemoCount };
}

describe('buildKeyTree', () => {
  it('nests slash-separated paths and marks leaves', () => {
    const tree = buildKeyTree([
      info('obs/state'),
      info('obs/rgb'),
      info('actions'),
    ]);

    // Groups sort before leaves; `obs` (group) precedes `actions` (leaf).
    expect(tree.map((node) => node.name)).toEqual(['obs', 'actions']);

    const [obs] = tree;
    expect(obs.keyInfo).toBeNull();
    // Children are sorted alphabetically for display...
    expect(obs.children.map((child) => child.name)).toEqual(['rgb', 'state']);
    // ...while leafKeyPaths preserves original insertion order.
    expect(obs.leafKeyPaths).toEqual(['obs/state', 'obs/rgb']);

    const [, actions] = tree;
    expect(actions.keyInfo).not.toBeNull();
    expect(actions.leafKeyPaths).toEqual(['actions']);
  });

  it('sorts leaves after groups and alphabetically within a level', () => {
    const tree = buildKeyTree([
      info('z_leaf'),
      info('group/b'),
      info('group/a'),
      info('a_leaf'),
    ]);
    expect(tree.map((node) => node.name)).toEqual([
      'group',
      'a_leaf',
      'z_leaf',
    ]);
    expect(tree[0].children.map((child) => child.name)).toEqual(['a', 'b']);
  });

  it('treats a path that is both a group and a leaf as a leaf', () => {
    // `obs` appears as a leaf and as a parent of `obs/state`.
    const tree = buildKeyTree([info('obs'), info('obs/state')]);
    const [obs] = tree;
    expect(obs.keyInfo).not.toBeNull();
    // Once a node carries keyInfo its leafKeyPaths is just its own path.
    expect(obs.leafKeyPaths).toEqual(['obs']);
  });
});

describe('sumKeyInfos', () => {
  it('merges duplicate paths and sums availability counts', () => {
    const summed = sumKeyInfos([
      info('actions', 2),
      info('obs/state', 3),
      info('actions', 5),
    ]);
    expect(summed).toEqual([
      { path: 'actions', availableInDemoCount: 7 },
      { path: 'obs/state', availableInDemoCount: 3 },
    ]);
  });

  it('does not mutate the input records', () => {
    const first = info('actions', 2);
    sumKeyInfos([first, info('actions', 5)]);
    expect(first.availableInDemoCount).toBe(2);
  });

  it('returns an empty array for empty input', () => {
    expect(sumKeyInfos([])).toEqual([]);
  });
});

describe('buildBackendKeyInfos', () => {
  it('returns an empty array for null input', () => {
    expect(buildBackendKeyInfos(null)).toEqual([]);
  });

  it('prefers keyCounts when present, sorted by path', () => {
    const result = buildBackendKeyInfos(
      fileInfo({ keyCounts: { 'obs/state': 3, actions: 4 } }),
    );
    expect(result).toEqual([
      { path: 'actions', availableInDemoCount: 4 },
      { path: 'obs/state', availableInDemoCount: 3 },
    ]);
  });

  it('falls back to keys with the demo count when keyCounts is absent', () => {
    const result = buildBackendKeyInfos(
      fileInfo({ keys: ['obs/state', 'actions'], demoCount: 4 }),
    );
    expect(result).toEqual([
      { path: 'actions', availableInDemoCount: 4 },
      { path: 'obs/state', availableInDemoCount: 4 },
    ]);
  });
});
