import { describe, expect, it } from 'vitest';

import {
  articulationFromRows,
  buildSlashTree,
  endEffectorRowsFromArticulation,
  formatAttrValue,
  formatGroupTitle,
  makeSegmentId,
  prettifyLeafValue,
  rowsFromArticulation,
  validateNamedRows,
} from './DatasetAttributesPage.utils';
import { type DatasetArticulation } from './python-backend';

describe('formatAttrValue', () => {
  it('returns a dash for null/undefined', () => {
    expect(formatAttrValue(null)).toBe('-');
    expect(formatAttrValue(undefined)).toBe('-');
  });

  it('returns strings unchanged', () => {
    expect(formatAttrValue('hello')).toBe('hello');
  });

  it('JSON-stringifies non-string values', () => {
    expect(formatAttrValue(42)).toBe('42');
    expect(formatAttrValue([1, 2])).toBe('[1,2]');
    expect(formatAttrValue({ a: 1 })).toBe('{"a":1}');
  });
});

describe('formatGroupTitle', () => {
  it('renders the root group specially', () => {
    expect(formatGroupTitle('/')).toBe('/.attrs');
    expect(formatGroupTitle('')).toBe('/.attrs');
  });

  it('strips a leading slash and appends .attrs', () => {
    expect(formatGroupTitle('/data')).toBe('data.attrs');
    expect(formatGroupTitle('/data/demo_0')).toBe('data/demo_0.attrs');
  });
});

describe('prettifyLeafValue', () => {
  it('returns a dash for empty/undefined input', () => {
    expect(prettifyLeafValue(undefined)).toBe('-');
    expect(prettifyLeafValue('')).toBe('-');
  });

  it('pretty-prints JSON objects and arrays', () => {
    expect(prettifyLeafValue('{"a":1}')).toBe('{\n  "a": 1\n}');
    expect(prettifyLeafValue('[1,2]')).toBe('[\n  1,\n  2\n]');
  });

  it('returns non-JSON and scalar-JSON values as-is', () => {
    expect(prettifyLeafValue('plain text')).toBe('plain text');
    expect(prettifyLeafValue('42')).toBe('42');
    expect(prettifyLeafValue('null')).toBe('null');
  });
});

describe('buildSlashTree', () => {
  it('nests slash-separated attribute names and sorts children', () => {
    const tree = buildSlashTree('/data', {
      'articulation/name': 'robot',
      'articulation/joint_number': 7,
      total: 12,
    });

    // Top level sorted alphabetically: `articulation` then `total`.
    expect(tree.map((node) => node.name)).toEqual(['articulation', 'total']);

    const articulation = tree[0];
    expect(articulation.children?.map((child) => child.name)).toEqual([
      'joint_number',
      'name',
    ]);
    // Leaf paths embed the group path and the dotted attribute name.
    expect(articulation.path).toBe('/data#articulation');
    const nameLeaf = articulation.children?.find(
      (child) => child.name === 'name',
    );
    expect(nameLeaf?.value).toBe('robot');
    // Each level appends `#<dotted-prefix>` onto the parent path.
    expect(nameLeaf?.path).toBe('/data#articulation#articulation/name');
  });

  it('formats scalar leaves via formatAttrValue', () => {
    const tree = buildSlashTree('/data', { total: 12, label: 'x' });
    const total = tree.find((node) => node.name === 'total');
    expect(total?.value).toBe('12');
    expect(total?.children).toBeUndefined();
  });

  it('ignores empty attribute names', () => {
    const tree = buildSlashTree('/g', { '': 'skip', keep: 1 });
    expect(tree.map((node) => node.name)).toEqual(['keep']);
  });
});

describe('makeSegmentId', () => {
  it('prefixes the id with the name and is unique across calls', () => {
    const a = makeSegmentId('arm');
    const b = makeSegmentId('arm');
    expect(a.startsWith('arm-')).toBe(true);
    expect(a).not.toBe(b);
  });
});

function articulation(
  overrides: Partial<DatasetArticulation>,
): DatasetArticulation {
  return {
    name: '',
    joint_number: null,
    segmentation: {},
    end_effectors: {},
    ...overrides,
  };
}

describe('rowsFromArticulation / endEffectorRowsFromArticulation', () => {
  it('unpacks segmentation into name-sorted rows', () => {
    const rows = rowsFromArticulation(
      articulation({
        segmentation: {
          gripper: { target: '[7:8]', obs: '[7:8]' },
          arm: { target: '[0:7]', obs: '[0:7]' },
        },
      }),
    );
    expect(rows.map((row) => row.name)).toEqual(['arm', 'gripper']);
    expect(rows[0]).toMatchObject({
      name: 'arm',
      target: '[0:7]',
      obs: '[0:7]',
    });
    expect(rows[0].id).toContain('arm');
  });

  it('unpacks end-effectors into name-sorted rows', () => {
    const rows = endEffectorRowsFromArticulation(
      articulation({
        end_effectors: {
          right: { pose: '[7:14]', gripper: '[14:15]' },
          left: { pose: '[0:7]', gripper: '[7:8]' },
        },
      }),
    );
    expect(rows.map((row) => row.name)).toEqual(['left', 'right']);
    expect(rows[0]).toMatchObject({
      name: 'left',
      pose: '[0:7]',
      gripper: '[7:8]',
    });
  });

  it('returns an empty array for empty maps', () => {
    expect(rowsFromArticulation(articulation({}))).toEqual([]);
    expect(endEffectorRowsFromArticulation(articulation({}))).toEqual([]);
  });
});

describe('articulationFromRows', () => {
  it('packs rows back into an articulation, trimming values', () => {
    const result = articulationFromRows(
      '  robot  ',
      '7',
      [{ id: '1', name: ' arm ', target: ' [0:7] ', obs: ' [0:7] ' }],
      [{ id: '2', name: ' left ', pose: ' [0:7] ', gripper: ' [7:8] ' }],
    );
    expect(result).toEqual({
      name: 'robot',
      joint_number: 7,
      segmentation: { arm: { target: '[0:7]', obs: '[0:7]' } },
      end_effectors: { left: { pose: '[0:7]', gripper: '[7:8]' } },
    });
  });

  it('drops rows whose name is blank after trimming', () => {
    const result = articulationFromRows(
      'robot',
      '5',
      [
        { id: '1', name: '  ', target: 'x', obs: 'y' },
        { id: '2', name: 'arm', target: 'a', obs: 'b' },
      ],
      [],
    );
    expect(Object.keys(result.segmentation)).toEqual(['arm']);
  });

  it('sets joint_number to null when not a finite integer', () => {
    expect(articulationFromRows('r', '', [], []).joint_number).toBeNull();
    expect(articulationFromRows('r', 'abc', [], []).joint_number).toBeNull();
  });
});

describe('validateNamedRows', () => {
  it('returns null when all names are present and unique', () => {
    expect(
      validateNamedRows([{ name: 'a' }, { name: 'b' }], 'segment'),
    ).toBeNull();
  });

  it('flags blank names', () => {
    expect(validateNamedRows([{ name: '  ' }], 'segment')).toBe(
      'Every segment needs a name before saving.',
    );
  });

  it('flags duplicate names (after trimming)', () => {
    expect(
      validateNamedRows([{ name: 'arm' }, { name: ' arm ' }], 'segment'),
    ).toBe('segment names must be unique: arm');
  });
});
