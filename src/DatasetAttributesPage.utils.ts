import { type DatasetArticulation } from './python-backend';

export interface SegmentRow {
  id: string;
  name: string;
  target: string;
  obs: string;
}

export interface EndEffectorRow {
  id: string;
  name: string;
  pose: string;
  gripper: string;
}

export interface AttributeTreeNode {
  name: string;
  path: string;
  value?: string;
  children?: AttributeTreeNode[];
}

/** Render an attribute value for display: strings as-is, everything else JSON. */
export function formatAttrValue(value: unknown): string {
  if (value === null || value === undefined) {
    return '-';
  }

  if (typeof value === 'string') {
    return value;
  }

  return JSON.stringify(value);
}

function createLeaf(
  path: string,
  name: string,
  value: unknown,
): AttributeTreeNode {
  return {
    path,
    name,
    value: formatAttrValue(value),
  };
}

/**
 * Build a tree from `/`-separated attribute names within a group. Leaf names may
 * themselves be nested (`a/b/c`), so this splits and merges shared prefixes.
 */
export function buildSlashTree(
  groupPath: string,
  attrs: Record<string, unknown>,
): AttributeTreeNode[] {
  interface MutableNode {
    name: string;
    path: string;
    value?: unknown;
    isLeaf: boolean;
    children: Map<string, MutableNode>;
  }

  const root: MutableNode = {
    name: '',
    path: groupPath,
    isLeaf: false,
    children: new Map(),
  };

  for (const [fullName, value] of Object.entries(attrs)) {
    const parts = fullName.split('/').filter((segment) => segment.length > 0);
    if (parts.length === 0) {
      continue;
    }

    let cursor = root;
    parts.forEach((segment, index) => {
      const isLeaf = index === parts.length - 1;
      let next = cursor.children.get(segment);
      if (!next) {
        next = {
          name: segment,
          path: `${cursor.path}#${parts.slice(0, index + 1).join('/')}`,
          isLeaf,
          children: new Map(),
        };
        cursor.children.set(segment, next);
      }
      if (isLeaf) {
        next.value = value;
        next.isLeaf = true;
      }
      cursor = next;
    });
  }

  function toNode(node: MutableNode): AttributeTreeNode {
    const sortedChildren = [...node.children.values()].sort((left, right) =>
      left.name.localeCompare(right.name),
    );

    if (sortedChildren.length === 0) {
      return createLeaf(node.path, node.name, node.value);
    }

    return {
      name: node.name,
      path: node.path,
      value: node.isLeaf ? formatAttrValue(node.value) : undefined,
      children: sortedChildren.map(toNode),
    };
  }

  return [...root.children.values()]
    .sort((left, right) => left.name.localeCompare(right.name))
    .map(toNode);
}

/** Title for a group's `.attrs` panel (`/` → `/.attrs`). */
export function formatGroupTitle(path: string): string {
  if (path === '/' || path === '') {
    return '/.attrs';
  }
  return `${path.replace(/^\//u, '')}.attrs`;
}

/** Pretty-print a leaf value, expanding JSON objects/arrays with indentation. */
export function prettifyLeafValue(formatted: string | undefined): string {
  if (formatted === undefined || formatted === '') {
    return '-';
  }
  try {
    const parsed: unknown = JSON.parse(formatted);
    if (parsed !== null && typeof parsed === 'object') {
      return JSON.stringify(parsed, null, 2);
    }
  } catch {
    // Not JSON — fall through and return as-is.
  }
  return formatted;
}

/** Generate a stable-enough unique row id from a name. */
export function makeSegmentId(name: string): string {
  return `${name}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

/** Unpack an articulation's segmentation map into editable, name-sorted rows. */
export function rowsFromArticulation(
  articulation: DatasetArticulation,
): SegmentRow[] {
  return Object.entries(articulation.segmentation)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, segment]) => ({
      id: makeSegmentId(name),
      name,
      target: segment.target,
      obs: segment.obs,
    }));
}

/** Unpack an articulation's end-effector map into editable, name-sorted rows. */
export function endEffectorRowsFromArticulation(
  articulation: DatasetArticulation,
): EndEffectorRow[] {
  return Object.entries(articulation.end_effectors)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, endEffector]) => ({
      id: makeSegmentId(name),
      name,
      pose: endEffector.pose,
      gripper: endEffector.gripper,
    }));
}

/** Pack editable rows back into a normalized articulation object. */
export function articulationFromRows(
  name: string,
  jointNumber: string,
  rows: SegmentRow[],
  endEffectorRows: EndEffectorRow[],
): DatasetArticulation {
  const segmentation: DatasetArticulation['segmentation'] = {};
  for (const row of rows) {
    const segmentName = row.name.trim();
    if (!segmentName) {
      continue;
    }

    segmentation[segmentName] = {
      target: row.target.trim(),
      obs: row.obs.trim(),
    };
  }
  const end_effectors: DatasetArticulation['end_effectors'] = {};
  for (const row of endEffectorRows) {
    const eefName = row.name.trim();
    if (!eefName) {
      continue;
    }

    end_effectors[eefName] = {
      pose: row.pose.trim(),
      gripper: row.gripper.trim(),
    };
  }

  const parsedJointNumber = Number.parseInt(jointNumber, 10);
  return {
    name: name.trim(),
    joint_number: Number.isFinite(parsedJointNumber) ? parsedJointNumber : null,
    segmentation,
    end_effectors,
  };
}

/** Return an error message if row names are blank or duplicated, else null. */
export function validateNamedRows(
  rows: { name: string }[],
  label: string,
): string | null {
  const names = new Set<string>();
  for (const row of rows) {
    const name = row.name.trim();
    if (!name) {
      return `Every ${label} needs a name before saving.`;
    }
    if (names.has(name)) {
      return `${label} names must be unique: ${name}`;
    }
    names.add(name);
  }
  return null;
}
