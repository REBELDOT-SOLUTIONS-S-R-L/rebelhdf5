import {
  createContext,
  startTransition,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react';
import { FiFile, FiFolder } from 'react-icons/fi';
import { Link, useSearchParams } from 'react-router-dom';

import {
  checkBackend,
  getDatasetAttributes,
  updateDatasetArticulation,
  type DatasetArticulation,
  type DatasetAttributeGroup,
  type DatasetAttributesResult,
  type PythonBackendStatus,
} from './python-backend';
import {
  getDatasetFeatureAvailability,
  useDatasetFeatureAvailability,
} from './feature-availability';
import {
  articulationFromRows,
  type AttributeTreeNode,
  buildSlashTree,
  endEffectorRowsFromArticulation,
  type EndEffectorRow,
  formatGroupTitle,
  makeSegmentId,
  prettifyLeafValue,
  rowsFromArticulation,
  type SegmentRow,
  validateNamedRows,
} from './DatasetAttributesPage.utils';
import { FileService, type H5File, useStore } from './stores';
import styles from './DatasetAttributesPage.module.css';

const EMPTY_ARTICULATION: DatasetArticulation = {
  name: '',
  joint_number: null,
  segmentation: {},
  end_effectors: {},
};

interface ExpandedLeafContextValue {
  isExpanded: (path: string) => boolean;
  toggle: (path: string) => void;
}

const ExpandedLeafContext = createContext<ExpandedLeafContextValue>({
  isExpanded: () => false,
  toggle: () => {},
});

function AttributeTreeItem({
  node,
  depth,
}: {
  node: AttributeTreeNode;
  depth: number;
}) {
  const { isExpanded, toggle } = useContext(ExpandedLeafContext);

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
          {node.value && (
            <small className={styles.treeMeta}>{node.value}</small>
          )}
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

  const expanded = isExpanded(node.path);
  const fullValue = expanded ? prettifyLeafValue(node.value) : null;

  return (
    <div className={styles.treeLeafWrapper}>
      <button
        type="button"
        className={`${styles.treeLeaf} ${expanded ? styles.treeLeafExpanded : ''}`}
        style={{ paddingLeft: `${depth * 1.1}rem` }}
        title={node.path}
        aria-expanded={expanded}
        onClick={() => {
          toggle(node.path);
        }}
      >
        <FiFile aria-hidden className={styles.treeLeafIcon} />
        <span className={styles.treeLeafLabel}>{node.name}</span>
        <small className={styles.treeMeta}>{node.value}</small>
      </button>
      {expanded && fullValue !== null && (
        <pre
          className={styles.treeLeafFull}
          style={{ paddingLeft: `${depth * 1.1 + 1.6}rem` }}
        >
          {fullValue}
        </pre>
      )}
    </div>
  );
}

function resolveActiveFile(
  opened: H5File[],
  fileUrl: string | null,
): H5File | null {
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
  const availability = useDatasetFeatureAvailability(opened);
  const attributesAvailability = getDatasetFeatureAvailability({
    file,
    feature: 'datasetAttributes',
    opened,
    availability,
  });

  const [backend, setBackend] = useState<PythonBackendStatus>({
    available: false,
    rootDir: null,
    version: null,
  });
  const [attributes, setAttributes] = useState<DatasetAttributesResult | null>(
    null,
  );
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
    checkBackend()
      .then((status) => {
        if (!cancelled) {
          setBackend(status);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setBackend({ available: false, rootDir: null, version: null });
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (
      !datasetPath ||
      !backend.available ||
      attributesAvailability.status !== 'available'
    ) {
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
        setJointNumber(
          articulation.joint_number == null
            ? ''
            : String(articulation.joint_number),
        );
        setSegments(rowsFromArticulation(articulation));
        setEndEffectors(endEffectorRowsFromArticulation(articulation));
        setLoading(false);
      })
      .catch((loadError: unknown) => {
        if (cancelled) {
          return;
        }

        setAttributes(null);
        setError(
          loadError instanceof Error ? loadError.message : String(loadError),
        );
        setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [attributesAvailability.status, backend.available, datasetPath]);

  const attributeGroups = useMemo<DatasetAttributeGroup[]>(() => {
    if (!attributes?.groups) {
      return [];
    }

    return attributes.groups
      .filter((group) => Object.keys(group.attrs).length > 0)
      .sort((left, right) => left.path.localeCompare(right.path));
  }, [attributes]);

  const [expandedLeaves, setExpandedLeaves] = useState<Set<string>>(new Set());
  const toggleExpandedLeaf = useCallback((path: string) => {
    setExpandedLeaves((current) => {
      const next = new Set(current);
      if (next.has(path)) {
        next.delete(path);
      } else {
        next.add(path);
      }
      return next;
    });
  }, []);
  const expandedContextValue = useMemo<ExpandedLeafContextValue>(
    () => ({
      isExpanded: (path) => expandedLeaves.has(path),
      toggle: toggleExpandedLeaf,
    }),
    [expandedLeaves, toggleExpandedLeaf],
  );

  function updateSegment(id: string, patch: Partial<SegmentRow>) {
    setSegments((current) =>
      current.map((segment) =>
        segment.id === id ? { ...segment, ...patch } : segment,
      ),
    );
  }

  function updateEndEffector(id: string, patch: Partial<EndEffectorRow>) {
    setEndEffectors((current) =>
      current.map((endEffector) =>
        endEffector.id === id ? { ...endEffector, ...patch } : endEffector,
      ),
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

    const validationError =
      validateNamedRows(segments, 'segment') ??
      validateNamedRows(endEffectors, 'end-effector');
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
      setJointNumber(
        articulation.joint_number == null
          ? ''
          : String(articulation.joint_number),
      );
      setSegments(rowsFromArticulation(articulation));
      setEndEffectors(endEffectorRowsFromArticulation(articulation));
      setSaveMessage('Articulation attributes saved.');
    } catch (saveError: unknown) {
      setError(
        saveError instanceof Error ? saveError.message : String(saveError),
      );
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
            Inspect attributes stored on /data and edit the articulation
            metadata.
          </p>
        </div>
      </header>

      {!file && (
        <section className={styles.panel}>
          <p className={styles.statusText}>
            Open an HDF5 file before editing dataset attributes.
          </p>
          <div className={styles.footerActions}>
            <Link className={styles.button} to="/">
              Open HDF5
            </Link>
          </div>
        </section>
      )}

      {file && file.service !== FileService.Local && (
        <section className={styles.panel}>
          <p className={styles.errorText}>
            Dataset attribute editing is available for local Electron files.
          </p>
        </section>
      )}

      {file && file.service === FileService.Local && !datasetPath && (
        <section className={styles.panel}>
          <p className={styles.errorText}>
            Reopen this file in the Electron app so the backend can access its
            filesystem path.
          </p>
        </section>
      )}

      {file && datasetPath && !backend.available && (
        <section className={styles.panel}>
          <p className={styles.errorText}>
            Python backend is not available. Restart the Electron app to enable
            metadata editing.
          </p>
        </section>
      )}

      {file &&
        datasetPath &&
        backend.available &&
        attributesAvailability.status === 'pending' && (
          <section className={styles.panel}>
            <p className={styles.statusText}>
              Inspecting dataset schema before loading attributes...
            </p>
          </section>
        )}

      {file &&
        datasetPath &&
        backend.available &&
        attributesAvailability.status === 'unavailable' && (
          <section className={styles.panel}>
            <p className={styles.errorText}>{attributesAvailability.reason}</p>
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
        <ExpandedLeafContext.Provider value={expandedContextValue}>
          {attributeGroups.map((group) => {
            const nodes = buildSlashTree(group.path, group.attrs);
            return (
              <section className={styles.panel} key={group.path}>
                <div className={styles.panelHeader}>
                  <h2 className={styles.panelTitle}>
                    {formatGroupTitle(group.path)}
                  </h2>
                  <p className={styles.statusText} title={group.path}>
                    {group.path}
                  </p>
                </div>
                <div className={styles.keyTree}>
                  {nodes.map((node) => (
                    <AttributeTreeItem key={node.path} node={node} depth={0} />
                  ))}
                </div>
              </section>
            );
          })}

          <section className={styles.panel}>
            <div className={styles.panelHeader}>
              <div>
                <h2 className={styles.panelTitle}>Articulation</h2>
                <p className={styles.statusText}>
                  Source: {attributes.articulationSource}
                </p>
              </div>
              {saveMessage && (
                <p className={styles.successText}>{saveMessage}</p>
              )}
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

          {segments.length > 0 && (
            <section className={styles.panel}>
              <div className={styles.panelHeader}>
                <h2 className={styles.panelTitle}>Segmentation Children</h2>
                <p className={styles.statusText}>
                  {segments.length} segment{segments.length === 1 ? '' : 's'}
                </p>
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
                <button
                  className={styles.secondaryButton}
                  type="button"
                  onClick={addSegment}
                >
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
                              updateSegment(segment.id, {
                                name: event.target.value,
                              });
                            }}
                          />
                        </td>
                        <td>
                          <input
                            className={styles.input}
                            placeholder="[x:y]"
                            value={segment.target}
                            onChange={(event) => {
                              updateSegment(segment.id, {
                                target: event.target.value,
                              });
                            }}
                          />
                        </td>
                        <td>
                          <input
                            className={styles.input}
                            placeholder="[x:y]"
                            value={segment.obs}
                            onChange={(event) => {
                              updateSegment(segment.id, {
                                obs: event.target.value,
                              });
                            }}
                          />
                        </td>
                        <td>
                          <div className={styles.segmentActions}>
                            <button
                              className={styles.dangerButton}
                              type="button"
                              onClick={() => {
                                setSegments((current) =>
                                  current.filter(
                                    (row) => row.id !== segment.id,
                                  ),
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
                <p className={styles.statusText}>
                  No segmentation children yet.
                </p>
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
          )}

          {endEffectors.length > 0 && (
            <section className={styles.panel}>
              <div className={styles.panelHeader}>
                <h2 className={styles.panelTitle}>End Effector Children</h2>
                <p className={styles.statusText}>
                  {endEffectors.length} end effector
                  {endEffectors.length === 1 ? '' : 's'}
                </p>
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
                <button
                  className={styles.secondaryButton}
                  type="button"
                  onClick={addEndEffector}
                >
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
                              updateEndEffector(endEffector.id, {
                                name: event.target.value,
                              });
                            }}
                          />
                        </td>
                        <td>
                          <input
                            className={styles.input}
                            placeholder="[x:y]"
                            value={endEffector.pose}
                            onChange={(event) => {
                              updateEndEffector(endEffector.id, {
                                pose: event.target.value,
                              });
                            }}
                          />
                        </td>
                        <td>
                          <input
                            className={styles.input}
                            placeholder="[x:y]"
                            value={endEffector.gripper}
                            onChange={(event) => {
                              updateEndEffector(endEffector.id, {
                                gripper: event.target.value,
                              });
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
                                  current.filter(
                                    (row) => row.id !== endEffector.id,
                                  ),
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
                <p className={styles.statusText}>
                  No end-effector children yet.
                </p>
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
          )}
        </ExpandedLeafContext.Provider>
      )}
    </div>
  );
}

export default DatasetAttributesPage;
