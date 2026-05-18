import { startTransition, useEffect, useMemo, useState } from 'react';
import { FiFile, FiFolder } from 'react-icons/fi';
import { Link, useSearchParams } from 'react-router-dom';

import {
  checkBackend,
  getDatasetAttributes,
  updateDatasetArticulation,
  type DatasetArticulation,
  type DatasetAttributesResult,
  type PythonBackendStatus,
} from './python-backend';
import { FileService, type H5File, useStore } from './stores';
import styles from './DatasetAttributesPage.module.css';

interface SegmentRow {
  id: string;
  name: string;
  target: string;
  obs: string;
}

interface EndEffectorRow {
  id: string;
  name: string;
  pose: string;
  gripper: string;
}

interface AttributeTreeNode {
  name: string;
  path: string;
  value?: string;
  children?: AttributeTreeNode[];
}

const EMPTY_ARTICULATION: DatasetArticulation = {
  name: '',
  joint_number: null,
  segmentation: {},
  end_effectors: {},
};

function formatAttrValue(value: unknown): string {
  if (value == null) {
    return '-';
  }

  if (typeof value === 'string') {
    return value;
  }

  return JSON.stringify(value);
}

function createLeaf(path: string, name: string, value: unknown): AttributeTreeNode {
  return {
    path,
    name,
    value: formatAttrValue(value),
  };
}

function buildAttributeTree(attributes: DatasetAttributesResult): AttributeTreeNode[] {
  const orderedAttrNames = [
    'schema_version',
    'fps',
    'env_args',
    'actions_frame',
    'num_episodes',
    'total_samples',
    'description',
  ];
  const usedNames = new Set([...orderedAttrNames, 'articulation']);
  const articulation = attributes.articulation ?? EMPTY_ARTICULATION;
  const segmentationChildren = Object.entries(articulation.segmentation ?? {})
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([segmentName, segment]) => ({
      name: segmentName,
      path: `articulation/segmentation/${segmentName}`,
      children: [
        createLeaf(
          `articulation/segmentation/${segmentName}/target`,
          'target',
          segment.target,
        ),
        createLeaf(
          `articulation/segmentation/${segmentName}/obs`,
          'obs',
          segment.obs,
        ),
      ],
    }));
  const endEffectorChildren = Object.entries(articulation.end_effectors ?? {})
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([eefName, endEffector]) => ({
      name: eefName,
      path: `articulation/end_effectors/${eefName}`,
      children: [
        createLeaf(
          `articulation/end_effectors/${eefName}/pose`,
          'pose',
          endEffector.pose,
        ),
        createLeaf(
          `articulation/end_effectors/${eefName}/gripper`,
          'gripper',
          endEffector.gripper,
        ),
      ],
    }));
  const knownNodes = orderedAttrNames.map((name) =>
    createLeaf(name, name, attributes.attrs[name]),
  );
  const otherNodes = Object.entries(attributes.attrs)
    .filter(([name]) => !usedNames.has(name))
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, value]) => createLeaf(name, name, value));

  return [
    ...knownNodes,
    {
      name: 'articulation',
      path: 'articulation',
      children: [
        createLeaf('articulation/name', 'name', articulation.name),
        createLeaf('articulation/joint_number', 'joint_number', articulation.joint_number),
        {
          name: 'segmentation',
          path: 'articulation/segmentation',
          value: segmentationChildren.length === 0 ? 'empty' : undefined,
          children: [
            createLeaf('articulation/segmentation/target', 'target', null),
            createLeaf('articulation/segmentation/obs', 'obs', null),
            ...segmentationChildren,
          ],
        },
        {
          name: 'end_effectors',
          path: 'articulation/end_effectors',
          value: endEffectorChildren.length === 0 ? 'empty' : undefined,
          children: endEffectorChildren,
        },
      ],
    },
    ...otherNodes,
  ];
}

function AttributeTreeItem({
  node,
  depth,
}: {
  node: AttributeTreeNode;
  depth: number;
}) {
  if (node.children) {
    return (
      <div className={styles.treeBranch}>
        <div
          className={styles.treeBranchRow}
          style={{ paddingLeft: `${depth * 1.1}rem` }}
          title={node.path}
        >
          <FiFolder aria-hidden className={styles.treeBranchIcon} />
          <span className={styles.treeBranchName}>{node.name}</span>
          {node.value && <small className={styles.treeMeta}>{node.value}</small>}
        </div>
        {node.children.length > 0 && (
          <div className={styles.treeChildren}>
            {node.children.map((child) => (
              <AttributeTreeItem
                key={child.path}
                node={child}
                depth={depth + 1}
              />
            ))}
          </div>
        )}
      </div>
    );
  }

  return (
    <div
      className={styles.treeLeaf}
      style={{ paddingLeft: `${depth * 1.1 + 0.75}rem` }}
      title={node.path}
    >
      <FiFile aria-hidden className={styles.treeLeafIcon} />
      <span className={styles.treeLeafLabel}>{node.name}</span>
      <small className={styles.treeMeta}>{node.value}</small>
    </div>
  );
}

function makeSegmentId(name: string): string {
  return `${name}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function rowsFromArticulation(articulation: DatasetArticulation): SegmentRow[] {
  return Object.entries(articulation.segmentation ?? {})
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, segment]) => ({
      id: makeSegmentId(name),
      name,
      target: segment.target,
      obs: segment.obs,
    }));
}

function endEffectorRowsFromArticulation(articulation: DatasetArticulation): EndEffectorRow[] {
  return Object.entries(articulation.end_effectors ?? {})
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, endEffector]) => ({
      id: makeSegmentId(name),
      name,
      pose: endEffector.pose,
      gripper: endEffector.gripper,
    }));
}

function articulationFromRows(
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

function validateNamedRows(
  rows: Array<{ name: string }>,
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

function resolveActiveFile(opened: H5File[], fileUrl: string | null): H5File | null {
  if (fileUrl) {
    return opened.find((file) => file.url === fileUrl) ?? null;
  }

  return opened[0] ?? null;
}

function DatasetAttributesPage() {
  const [searchParams] = useSearchParams();
  const opened = useStore((state) => state.opened);
  const file = resolveActiveFile(opened, searchParams.get('url'));
  const datasetPath = file?.serverPath ?? null;

  const [backend, setBackend] = useState<PythonBackendStatus>({
    available: false,
    rootDir: null,
    version: null,
  });
  const [attributes, setAttributes] = useState<DatasetAttributesResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);
  const [robotName, setRobotName] = useState('');
  const [jointNumber, setJointNumber] = useState('');
  const [segments, setSegments] = useState<SegmentRow[]>([]);
  const [endEffectors, setEndEffectors] = useState<EndEffectorRow[]>([]);
  const [newSegmentName, setNewSegmentName] = useState('');
  const [newEndEffectorName, setNewEndEffectorName] = useState('');

  useEffect(() => {
    let cancelled = false;
    checkBackend().then((status) => {
      if (!cancelled) {
        setBackend(status);
      }
    }).catch(() => {
      if (!cancelled) {
        setBackend({ available: false, rootDir: null, version: null });
      }
    });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!datasetPath || !backend.available) {
      setAttributes(null);
      setLoading(false);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);
    setSaveMessage(null);

    getDatasetAttributes(datasetPath)
      .then((result) => {
        if (cancelled) {
          return;
        }

        const articulation = result.articulation ?? EMPTY_ARTICULATION;
        setAttributes(result);
        setRobotName(articulation.name);
        setJointNumber(articulation.joint_number == null ? '' : String(articulation.joint_number));
        setSegments(rowsFromArticulation(articulation));
        setEndEffectors(endEffectorRowsFromArticulation(articulation));
        setLoading(false);
      })
      .catch((loadError: unknown) => {
        if (cancelled) {
          return;
        }

        setAttributes(null);
        setError(loadError instanceof Error ? loadError.message : String(loadError));
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [backend.available, datasetPath]);

  const attributeTree = useMemo(
    () => (attributes ? buildAttributeTree(attributes) : []),
    [attributes],
  );

  function updateSegment(id: string, patch: Partial<SegmentRow>) {
    setSegments((current) =>
      current.map((segment) => (segment.id === id ? { ...segment, ...patch } : segment)),
    );
  }

  function updateEndEffector(id: string, patch: Partial<EndEffectorRow>) {
    setEndEffectors((current) =>
      current.map((endEffector) => (endEffector.id === id ? { ...endEffector, ...patch } : endEffector)),
    );
  }

  function addSegment() {
    const name = newSegmentName.trim();
    if (!name) {
      setError('Enter a segment name before adding it.');
      return;
    }

    if (segments.some((segment) => segment.name.trim() === name)) {
      setError(`Segment already exists: ${name}`);
      return;
    }

    startTransition(() => {
      setSegments((current) => [
        ...current,
        { id: makeSegmentId(name), name, target: '', obs: '' },
      ]);
      setNewSegmentName('');
      setError(null);
      setSaveMessage(null);
    });
  }

  function addEndEffector() {
    const name = newEndEffectorName.trim();
    if (!name) {
      setError('Enter an end-effector name before adding it.');
      return;
    }

    if (endEffectors.some((endEffector) => endEffector.name.trim() === name)) {
      setError(`End-effector already exists: ${name}`);
      return;
    }

    startTransition(() => {
      setEndEffectors((current) => [
        ...current,
        { id: makeSegmentId(name), name, pose: '', gripper: '' },
      ]);
      setNewEndEffectorName('');
      setError(null);
      setSaveMessage(null);
    });
  }

  async function saveArticulation() {
    if (!datasetPath) {
      setError('This file does not expose an editable local filesystem path.');
      return;
    }

    const validationError = validateNamedRows(segments, 'segment')
      ?? validateNamedRows(endEffectors, 'end-effector');
    if (validationError) {
      setError(validationError);
      return;
    }

    setSaving(true);
    setError(null);
    setSaveMessage(null);

    try {
      const result = await updateDatasetArticulation(
        datasetPath,
        articulationFromRows(robotName, jointNumber, segments, endEffectors),
      );
      const articulation = result.articulation ?? EMPTY_ARTICULATION;
      setAttributes(result);
      setRobotName(articulation.name);
      setJointNumber(articulation.joint_number == null ? '' : String(articulation.joint_number));
      setSegments(rowsFromArticulation(articulation));
      setEndEffectors(endEffectorRowsFromArticulation(articulation));
      setSaveMessage('Articulation attributes saved.');
    } catch (saveError: unknown) {
      setError(saveError instanceof Error ? saveError.message : String(saveError));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className={styles.root}>
      <header className={styles.header}>
        <div>
          <p className={styles.eyebrow}>Metadata</p>
          <h1 className={styles.title}>Dataset Attributes</h1>
          <p className={styles.subtitle}>
            Inspect attributes stored on /data and edit the articulation metadata.
          </p>
        </div>
      </header>

      {!file && (
        <section className={styles.panel}>
          <p className={styles.statusText}>Open an HDF5 file before editing dataset attributes.</p>
          <div className={styles.footerActions}>
            <Link className={styles.button} to="/">Open HDF5</Link>
          </div>
        </section>
      )}

      {file && file.service !== FileService.Local && (
        <section className={styles.panel}>
          <p className={styles.errorText}>Dataset attribute editing is available for local Electron files.</p>
        </section>
      )}

      {file && file.service === FileService.Local && !datasetPath && (
        <section className={styles.panel}>
          <p className={styles.errorText}>Reopen this file in the Electron app so the backend can access its filesystem path.</p>
        </section>
      )}

      {file && datasetPath && !backend.available && (
        <section className={styles.panel}>
          <p className={styles.errorText}>Python backend is not available. Restart the Electron app to enable metadata editing.</p>
        </section>
      )}

      {loading && (
        <section className={styles.panel}>
          <p className={styles.statusText}>Loading dataset attributes...</p>
        </section>
      )}

      {error && (
        <section className={styles.panel}>
          <p className={styles.errorText}>{error}</p>
        </section>
      )}

      {attributes && (
        <>
          <section className={styles.panel}>
            <div className={styles.panelHeader}>
              <h2 className={styles.panelTitle}>/data Attributes</h2>
              <p className={styles.statusText}>{file?.name}</p>
            </div>
            <div className={styles.keyTree}>
              {attributeTree.map((node) => (
                <AttributeTreeItem key={node.path} node={node} depth={0} />
              ))}
            </div>
          </section>

          <section className={styles.panel}>
            <div className={styles.panelHeader}>
              <div>
                <h2 className={styles.panelTitle}>Articulation Segmentation</h2>
                <p className={styles.statusText}>Source: {attributes.articulationSource}</p>
              </div>
              {saveMessage && <p className={styles.successText}>{saveMessage}</p>}
            </div>

            <div className={styles.fieldGrid}>
              <label className={styles.field}>
                <span className={styles.label}>Robot name</span>
                <input
                  className={styles.input}
                  value={robotName}
                  onChange={(event) => {
                    setRobotName(event.target.value);
                  }}
                />
              </label>
              <label className={styles.field}>
                <span className={styles.label}>Joint number</span>
                <input
                  className={styles.input}
                  type="number"
                  min={0}
                  step={1}
                  value={jointNumber}
                  onChange={(event) => {
                    setJointNumber(event.target.value);
                  }}
                />
              </label>
            </div>
          </section>

          <section className={styles.panel}>
            <div className={styles.panelHeader}>
              <h2 className={styles.panelTitle}>Segmentation Children</h2>
              <p className={styles.statusText}>{segments.length} segment{segments.length === 1 ? '' : 's'}</p>
            </div>

            <div className={styles.addRow}>
              <input
                className={styles.input}
                placeholder="segment_name"
                value={newSegmentName}
                onChange={(event) => {
                  setNewSegmentName(event.target.value);
                }}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    event.preventDefault();
                    addSegment();
                  }
                }}
              />
              <button className={styles.secondaryButton} type="button" onClick={addSegment}>
                Add Segment
              </button>
            </div>

            {segments.length > 0 ? (
              <table className={styles.segmentTable}>
                <thead>
                  <tr>
                    <th>Segment</th>
                    <th>Target</th>
                    <th>Obs</th>
                    <th aria-label="Actions" />
                  </tr>
                </thead>
                <tbody>
                  {segments.map((segment) => (
                    <tr key={segment.id}>
                      <td>
                        <input
                          className={styles.input}
                          value={segment.name}
                          onChange={(event) => {
                            updateSegment(segment.id, { name: event.target.value });
                          }}
                        />
                      </td>
                      <td>
                        <input
                          className={styles.input}
                          placeholder="[x:y]"
                          value={segment.target}
                          onChange={(event) => {
                            updateSegment(segment.id, { target: event.target.value });
                          }}
                        />
                      </td>
                      <td>
                        <input
                          className={styles.input}
                          placeholder="[x:y]"
                          value={segment.obs}
                          onChange={(event) => {
                            updateSegment(segment.id, { obs: event.target.value });
                          }}
                        />
                      </td>
                      <td>
                        <div className={styles.segmentActions}>
                          <button
                            className={styles.dangerButton}
                            type="button"
                            onClick={() => {
                              setSegments((current) => current.filter((row) => row.id !== segment.id));
                            }}
                          >
                            Remove
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className={styles.statusText}>No segmentation children yet.</p>
            )}

            <div className={styles.footerActions}>
              <button
                className={styles.button}
                type="button"
                onClick={() => {
                  void saveArticulation();
                }}
                disabled={saving}
              >
                {saving ? 'Saving...' : 'Save Articulation'}
              </button>
            </div>
          </section>

          <section className={styles.panel}>
            <div className={styles.panelHeader}>
              <h2 className={styles.panelTitle}>End Effector Children</h2>
              <p className={styles.statusText}>{endEffectors.length} end effector{endEffectors.length === 1 ? '' : 's'}</p>
            </div>

            <div className={styles.addRow}>
              <input
                className={styles.input}
                placeholder="eef_name"
                value={newEndEffectorName}
                onChange={(event) => {
                  setNewEndEffectorName(event.target.value);
                }}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') {
                    event.preventDefault();
                    addEndEffector();
                  }
                }}
              />
              <button className={styles.secondaryButton} type="button" onClick={addEndEffector}>
                Add End Effector
              </button>
            </div>

            {endEffectors.length > 0 ? (
              <table className={styles.segmentTable}>
                <thead>
                  <tr>
                    <th>End Effector</th>
                    <th>Pose</th>
                    <th>Gripper</th>
                    <th aria-label="Actions" />
                  </tr>
                </thead>
                <tbody>
                  {endEffectors.map((endEffector) => (
                    <tr key={endEffector.id}>
                      <td>
                        <input
                          className={styles.input}
                          value={endEffector.name}
                          onChange={(event) => {
                            updateEndEffector(endEffector.id, { name: event.target.value });
                          }}
                        />
                      </td>
                      <td>
                        <input
                          className={styles.input}
                          placeholder="[x:y]"
                          value={endEffector.pose}
                          onChange={(event) => {
                            updateEndEffector(endEffector.id, { pose: event.target.value });
                          }}
                        />
                      </td>
                      <td>
                        <input
                          className={styles.input}
                          placeholder="[x:y]"
                          value={endEffector.gripper}
                          onChange={(event) => {
                            updateEndEffector(endEffector.id, { gripper: event.target.value });
                          }}
                        />
                      </td>
                      <td>
                        <div className={styles.segmentActions}>
                          <button
                            className={styles.dangerButton}
                            type="button"
                            onClick={() => {
                              setEndEffectors((current) =>
                                current.filter((row) => row.id !== endEffector.id),
                              );
                            }}
                          >
                            Remove
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <p className={styles.statusText}>No end-effector children yet.</p>
            )}

            <div className={styles.footerActions}>
              <button
                className={styles.button}
                type="button"
                onClick={() => {
                  void saveArticulation();
                }}
                disabled={saving}
              >
                {saving ? 'Saving...' : 'Save Articulation'}
              </button>
            </div>
          </section>
        </>
      )}
    </div>
  );
}

export default DatasetAttributesPage;
