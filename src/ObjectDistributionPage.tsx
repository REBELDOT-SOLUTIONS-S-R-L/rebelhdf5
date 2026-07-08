import { useEffect, useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';

import { buildFailureAnalysis } from './pose-trace/failureAnalysis';
import { loadObjectDistribution, openPoseTraceSource } from './pose-trace/hdf5';
import Plot from './pose-trace/PlotlyChart';
import {
  buildObjectDistributionData,
  buildObjectDistributionLayout,
  buildEmptyLayout,
  buildFailureMapData,
  buildFailureMapLayout,
  buildFailureSliceData,
  buildFailureSliceLayout,
} from './pose-trace/plotConfig';
import {
  DEFAULT_OBJECT_DISTRIBUTION_ANCHOR,
  type ObjectDistributionPoint,
  type ObjectDistributionResult,
  type PoseTraceSource,
} from './pose-trace/types';
import {
  analysisTitle,
  formatPercent,
  formatRange,
  formatSupport,
  hashRevisionKey,
  type ObjectViewTab,
  type PlotClickEvent,
  resolveClickedPoint,
  selectDatasetGroup,
} from './ObjectDistributionPage.utils';
import { type H5File, useStore } from './stores';
import styles from './ObjectDistributionPage.module.css';

interface SourceState {
  source: PoseTraceSource | null;
  loading: boolean;
  error: string | null;
}

const VIEW_TABS: Array<{ id: ObjectViewTab; label: string }> = [
  { id: 'scatter', label: 'Scatter' },
  { id: 'position', label: 'Position Map' },
  { id: 'rotation', label: 'Rotation Map' },
  { id: 'slices', label: 'Spatial Slices' },
];

const MIN_GENERATED_COUNT_OPTIONS = [1, 3, 5, 10];

function usePoseSource(file: H5File | null): SourceState {
  const [state, setState] = useState<SourceState>({
    source: null,
    loading: false,
    error: null,
  });

  useEffect(() => {
    let cancelled = false;
    let cleanup = () => {};

    if (!file) {
      setState({ source: null, loading: false, error: null });
      return () => {
        cleanup();
      };
    }

    setState({ source: null, loading: true, error: null });

    openPoseTraceSource(file)
      .then((source) => {
        if (cancelled) {
          source.cleanup();
          return;
        }

        cleanup = source.cleanup;
        setState({ source, loading: false, error: null });
      })
      .catch((error: unknown) => {
        if (cancelled) {
          return;
        }

        setState({
          source: null,
          loading: false,
          error: error instanceof Error ? error.message : String(error),
        });
      });

    return () => {
      cancelled = true;
      cleanup();
    };
  }, [file]);

  return state;
}

function EmptyState({ openedFileCount }: { openedFileCount: number }) {
  return (
    <div className={styles.emptyState}>
      <h2 className={styles.emptyTitle}>Object Distribution</h2>
      <p className={styles.emptyText}>
        Open HDF5 files in rebelHDF5, then use this page to compare generated datasets, inspect
        source-demo links, and analyze failure coverage across the object reset space.
      </p>
      <div className={styles.emptyActions}>
        <Link className={styles.openBtn} to="/">
          Open HDF5
        </Link>
        {openedFileCount > 0 && (
          <span>{openedFileCount} opened file{openedFileCount === 1 ? '' : 's'} available in the sidebar.</span>
        )}
      </div>
    </div>
  );
}

function ObjectDistributionPage() {
  const opened = useStore((state) => state.opened);
  const [searchParams] = useSearchParams();
  const activeUrl = searchParams.get('url');

  const [successUrl, setSuccessUrl] = useState<string | null>(null);
  const [failedUrl, setFailedUrl] = useState<string | null>(null);
  const [teleopUrl, setTeleopUrl] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<ObjectViewTab>('scatter');
  const anchor = DEFAULT_OBJECT_DISTRIBUTION_ANCHOR;
  const [objectName, setObjectName] = useState<string | null>(null);
  const [showTeleopOverlay, setShowTeleopOverlay] = useState(true);
  const [minGeneratedCount, setMinGeneratedCount] = useState(3);
  const [result, setResult] = useState<ObjectDistributionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selectedPoint, setSelectedPoint] = useState<ObjectDistributionPoint | null>(null);

  useEffect(() => {
    const availableUrls = new Set(opened.map((file) => file.url));
    const selection = selectDatasetGroup(opened, activeUrl);

    setSuccessUrl((current) => (current && availableUrls.has(current) ? current : selection.successUrl));
    setFailedUrl((current) => (current && availableUrls.has(current) ? current : selection.failedUrl));
    setTeleopUrl((current) => (current && availableUrls.has(current) ? current : selection.teleopUrl));
  }, [activeUrl, opened]);

  const successFile = useMemo(
    () => opened.find((file) => file.url === successUrl) ?? null,
    [opened, successUrl],
  );
  const failedFile = useMemo(
    () => opened.find((file) => file.url === failedUrl) ?? null,
    [failedUrl, opened],
  );
  const teleopFile = useMemo(
    () => opened.find((file) => file.url === teleopUrl) ?? null,
    [opened, teleopUrl],
  );

  const successState = usePoseSource(successFile);
  const failedState = usePoseSource(failedFile);
  const teleopState = usePoseSource(teleopFile);

  const selectedStates = useMemo(
    () => [
      { label: 'Successful Generated Dataset', url: successUrl, state: successState },
      { label: 'Failed Generated Dataset', url: failedUrl, state: failedState },
      { label: 'Teleop Dataset', url: teleopUrl, state: teleopState },
    ].filter((entry) => entry.url),
    [failedState, failedUrl, successState, successUrl, teleopState, teleopUrl],
  );

  useEffect(() => {
    const sourceErrors = selectedStates
      .filter((entry) => entry.state.error)
      .map((entry) => `${entry.label}: ${entry.state.error}`);
    if (sourceErrors.length > 0) {
      setLoadError(sourceErrors.join(' | '));
      setLoading(false);
      setResult(null);
      setSelectedPoint(null);
      return;
    }

    if (selectedStates.length === 0) {
      setLoadError(null);
      setLoading(false);
      setResult(null);
      setSelectedPoint(null);
      return;
    }

    if (selectedStates.some((entry) => entry.state.loading || !entry.state.source)) {
      setLoading(true);
      setLoadError(null);
      setResult(null);
      setSelectedPoint(null);
      return;
    }

    let cancelled = false;
    setLoading(true);
    setLoadError(null);
    setResult(null);
    setSelectedPoint(null);

    void loadObjectDistribution({
      successSourceId: successState.source?.sourceId ?? null,
      failedSourceId: failedState.source?.sourceId ?? null,
      teleopSourceId: teleopState.source?.sourceId ?? null,
      anchor,
      objectName,
    })
      .then((nextResult) => {
        if (cancelled) {
          return;
        }

        setResult(nextResult);
        setSelectedPoint(null);
      })
      .catch((error: unknown) => {
        if (cancelled) {
          return;
        }

        setResult(null);
        setLoadError(error instanceof Error ? error.message : String(error));
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [
    anchor,
    failedState.error,
    failedState.loading,
    failedState.source,
    objectName,
    selectedStates,
    successState.error,
    successState.loading,
    successState.source,
    teleopState.error,
    teleopState.loading,
    teleopState.source,
  ]);

  const availableObjects = useMemo(() => result?.availableObjects ?? [], [result]);

  // Drop a stale selection when the chosen object is no longer present in the
  // selected datasets (falls back to the default first-object behavior).
  useEffect(() => {
    if (objectName && availableObjects.length > 0 && !availableObjects.includes(objectName)) {
      setObjectName(null);
    }
  }, [availableObjects, objectName]);

  const totalPointCount = (result?.successPoints.length ?? 0)
    + (result?.failedPoints.length ?? 0)
    + (result?.teleopPoints.length ?? 0);
  const hasScatterData = totalPointCount > 0;
  const hasAnalysisDatasets = Boolean(successUrl && failedUrl);
  const analysisResult = useMemo(
    () => (result ? buildFailureAnalysis(result, { minGeneratedCount }) : null),
    [minGeneratedCount, result],
  );

  const analysisMessage = useMemo(() => {
    if (loading) {
      return 'Loading object-distribution data…';
    }

    if (!hasAnalysisDatasets) {
      return 'Select both a successful and a failed generated dataset to render failure analysis.';
    }

    if (!analysisResult) {
      return 'No generated episodes with a complete initial reset pose were found in the selected datasets.';
    }

    return '';
  }, [analysisResult, hasAnalysisDatasets, loading]);

  const teleopMessage = useMemo(() => {
    if (
      loading
      || loadError
      || !teleopUrl
      || !teleopState.source
      || !result
      || result.teleopPoints.length > 0
    ) {
      return '';
    }

    const diagnostics = result.teleopDiagnostics;
    if (!diagnostics || diagnostics.totalDemos === 0) {
      return '';
    }

    if (diagnostics.missingAnchorCount === diagnostics.totalDemos) {
      return 'Selected teleop dataset does not expose initial_state/rigid_objects/<name>/initial_pose, so it cannot be plotted.';
    }

    return `Selected teleop dataset contributed 0 of ${diagnostics.totalDemos} demos for the initial pose.`;
  }, [loadError, loading, result, teleopState.source, teleopUrl]);

  const plotRevision = useMemo(
    () => hashRevisionKey([
      activeTab,
      anchor,
      objectName ?? 'default',
      successState.source?.sourceId ?? 'none',
      failedState.source?.sourceId ?? 'none',
      teleopState.source?.sourceId ?? 'none',
      showTeleopOverlay,
      minGeneratedCount,
    ]),
    [
      activeTab,
      anchor,
      failedState.source,
      minGeneratedCount,
      objectName,
      showTeleopOverlay,
      successState.source,
      teleopState.source,
    ],
  );

  const plotData = useMemo(() => {
    if (activeTab === 'scatter') {
      return buildObjectDistributionData(result, selectedPoint);
    }

    if (!analysisResult) {
      return [];
    }

    if (activeTab === 'position') {
      return buildFailureMapData(analysisResult.positionMap, showTeleopOverlay);
    }

    if (activeTab === 'rotation') {
      return buildFailureMapData(analysisResult.rotationMap, showTeleopOverlay);
    }

    return buildFailureSliceData(analysisResult.slices, showTeleopOverlay);
  }, [activeTab, analysisResult, result, selectedPoint, showTeleopOverlay]);

  const plotLayout = useMemo(() => {
    if (activeTab === 'scatter') {
      if (hasScatterData) {
        return buildObjectDistributionLayout(result, anchor);
      }

      return buildEmptyLayout(
        'Object Distribution',
        loading
          ? 'Loading object-distribution data…'
          : 'Select dataset files to render the initial-pose scatter plot.',
      );
    }

    if (!analysisResult) {
      return buildEmptyLayout(analysisTitle(activeTab), analysisMessage);
    }

    if (activeTab === 'position') {
      return buildFailureMapLayout(analysisResult.positionMap);
    }

    if (activeTab === 'rotation') {
      return buildFailureMapLayout(analysisResult.rotationMap);
    }

    return buildFailureSliceLayout(analysisResult.slices);
  }, [activeTab, analysisMessage, analysisResult, anchor, hasScatterData, loading, result]);

  return (
    <div className={styles.root}>
      <header className={styles.header}>
        <div>
          <p className={styles.eyebrow}>Distribution</p>
          <h1 className={styles.title}>Object Distribution</h1>
          <p className={styles.subtitle}>
            Compare generated datasets and map failure coverage over the object initial-pose reset
            space. Use the analysis tabs to find under-covered teleop regions.
          </p>
        </div>
      </header>

      {opened.length === 0 && <EmptyState openedFileCount={opened.length} />}

      {opened.length > 0 && (
        <>
          <section className={styles.controlsCard}>
            <div className={styles.tabBar} role="tablist" aria-label="Object distribution views">
              {VIEW_TABS.map((tab) => (
                <button
                  key={tab.id}
                  type="button"
                  role="tab"
                  aria-selected={activeTab === tab.id}
                  className={activeTab === tab.id ? styles.tabBtnActive : styles.tabBtn}
                  onClick={() => {
                    setActiveTab(tab.id);
                  }}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            <div className={styles.controlGrid}>
              <div className={styles.field}>
                <label className={styles.fieldLabel} htmlFor="object-success-source">
                  Successful Generated Dataset
                </label>
                <select
                  id="object-success-source"
                  className={styles.select}
                  value={successUrl ?? ''}
                  onChange={(event) => {
                    setSuccessUrl(event.target.value || null);
                  }}
                >
                  <option value="">None</option>
                  {opened.map((file) => (
                    <option key={file.url} value={file.url}>
                      {file.name}
                    </option>
                  ))}
                </select>
              </div>

              <div className={styles.field}>
                <label className={styles.fieldLabel} htmlFor="object-failed-source">
                  Failed Generated Dataset
                </label>
                <select
                  id="object-failed-source"
                  className={styles.select}
                  value={failedUrl ?? ''}
                  onChange={(event) => {
                    setFailedUrl(event.target.value || null);
                  }}
                >
                  <option value="">None</option>
                  {opened.map((file) => (
                    <option key={file.url} value={file.url}>
                      {file.name}
                    </option>
                  ))}
                </select>
              </div>

              <div className={styles.field}>
                <label className={styles.fieldLabel} htmlFor="object-teleop-source">
                  Teleop Dataset
                </label>
                <select
                  id="object-teleop-source"
                  className={styles.select}
                  value={teleopUrl ?? ''}
                  onChange={(event) => {
                    setTeleopUrl(event.target.value || null);
                  }}
                >
                  <option value="">None</option>
                  {opened.map((file) => (
                    <option key={file.url} value={file.url}>
                      {file.name}
                    </option>
                  ))}
                </select>
              </div>

              {activeTab === 'scatter' ? (
                <div className={styles.field}>
                  <label className={styles.fieldLabel} htmlFor="object-object">
                    Reference Object
                  </label>
                  <select
                    id="object-object"
                    className={styles.select}
                    value={objectName ?? (availableObjects.length > 0 ? availableObjects[0] : '')}
                    disabled={availableObjects.length === 0}
                    onChange={(event) => {
                      setObjectName(event.target.value || null);
                    }}
                  >
                    {availableObjects.length === 0 && (
                      <option value="">No rigid objects found</option>
                    )}
                    {availableObjects.map((name) => (
                      <option key={name} value={name}>
                        {name}
                      </option>
                    ))}
                  </select>
                </div>
              ) : (
                <>
                  <div className={styles.field}>
                    <label className={styles.fieldLabel} htmlFor="object-min-support">
                      Min Neighborhood Support
                    </label>
                    <select
                      id="object-min-support"
                      className={styles.select}
                      value={String(minGeneratedCount)}
                      onChange={(event) => {
                        setMinGeneratedCount(Number(event.target.value));
                      }}
                    >
                      {MIN_GENERATED_COUNT_OPTIONS.map((value) => (
                        <option key={value} value={value}>
                          {value}+
                        </option>
                      ))}
                    </select>
                  </div>

                  <label className={styles.checkboxField} htmlFor="object-show-teleop">
                    <input
                      id="object-show-teleop"
                      type="checkbox"
                      checked={showTeleopOverlay}
                      onChange={(event) => {
                        setShowTeleopOverlay(event.target.checked);
                      }}
                    />
                    <span>Show teleop overlay</span>
                  </label>
                </>
              )}
            </div>

            <div className={styles.statusRow}>
              <div className={styles.statusItem}>
                <span className={styles.statusKey}>Opened:</span> {opened.length}
              </div>
              <div className={styles.statusItem}>
                <span className={styles.statusKey}>Success:</span> {result?.successPoints.length ?? 0}
              </div>
              <div className={styles.statusItem}>
                <span className={styles.statusKey}>Failed:</span> {result?.failedPoints.length ?? 0}
              </div>
              <div className={styles.statusItem}>
                <span className={styles.statusKey}>Teleop:</span> {result?.teleopPoints.length ?? 0}
              </div>

              {activeTab !== 'scatter' && (
                <>
                  <div className={styles.statusItem}>
                    <span className={styles.statusKey}>Analyzed Generated:</span>{' '}
                    {analysisResult?.stats.analyzedGeneratedCount ?? 0}
                  </div>
                  <div className={styles.statusItem}>
                    <span className={styles.statusKey}>Skipped Generated:</span>{' '}
                    {analysisResult?.stats.skippedGeneratedCount ?? 0}
                  </div>
                  <div className={styles.statusItem}>
                    <span className={styles.statusKey}>Analyzed Teleop:</span>{' '}
                    {analysisResult?.stats.analyzedTeleopCount ?? 0}
                  </div>
                  <div className={styles.statusItem}>
                    <span className={styles.statusKey}>Min Support:</span> {minGeneratedCount}+
                  </div>
                </>
              )}
            </div>
          </section>

          {(loadError || loading || teleopMessage) && (
            <section className={styles.messageCard}>
              {loadError && <p className={styles.errorText}>{loadError}</p>}
              {loading && <p className={styles.infoText}>Loading object-distribution data…</p>}
              {!loadError && !loading && teleopMessage && <p className={styles.infoText}>{teleopMessage}</p>}
            </section>
          )}

          <section className={styles.chartCard}>
            <Plot
              revision={plotRevision}
              data={plotData}
              layout={plotLayout}
              onClick={activeTab === 'scatter'
                ? (event) => {
                    setSelectedPoint(resolveClickedPoint(event as PlotClickEvent, result));
                  }
                : undefined}
              onDoubleClick={activeTab === 'scatter'
                ? () => {
                    setSelectedPoint(null);
                    return false;
                  }
                : undefined}
              useResizeHandler
              style={{ width: '100%' }}
              config={{ responsive: true }}
            />
          </section>

          {activeTab !== 'scatter' && (
            <section className={styles.recommendationsCard}>
              <div className={styles.recommendationsHeader}>
                <div>
                  <p className={styles.recommendationsEyebrow}>Recommendations</p>
                  <h2 className={styles.recommendationsTitle}>Teleop Candidate Regions</h2>
                  <p className={styles.recommendationsText}>
                    Ranked from connected high-risk neighborhoods in the spatial slices, using
                    failure confidence, local support, and nearby teleop density.
                  </p>
                </div>
              </div>

              {!analysisResult && (
                <p className={styles.infoText}>{analysisMessage}</p>
              )}

              {analysisResult && analysisResult.recommendations.length === 0 && (
                <p className={styles.infoText}>
                  No connected high-risk regions met the current neighborhood-support threshold.
                </p>
              )}

              {analysisResult && analysisResult.recommendations.length > 0 && (
                <div className={styles.recommendationsGrid}>
                  {analysisResult.recommendations.map((recommendation, index) => (
                    <article key={`${recommendation.sliceRowIndex}-${recommendation.sliceColIndex}-${index}`} className={styles.recommendationItem}>
                      <div className={styles.recommendationRank}>#{index + 1}</div>
                      <p className={styles.recommendationRate}>
                        {formatPercent(recommendation.failureRate)} local fail
                      </p>
                      <p className={styles.recommendationLine}>
                        <span className={styles.recommendationLabel}>95% ci</span>
                        {formatPercent(recommendation.confidenceLower)} to {formatPercent(recommendation.confidenceUpper)}
                      </p>
                      <p className={styles.recommendationLine}>
                        <span className={styles.recommendationLabel}>rot x</span>
                        {formatRange(recommendation.rotXStart, recommendation.rotXEnd, 'deg')}
                      </p>
                      <p className={styles.recommendationLine}>
                        <span className={styles.recommendationLabel}>rot y</span>
                        {formatRange(recommendation.rotYStart, recommendation.rotYEnd, 'deg')}
                      </p>
                      <p className={styles.recommendationLine}>
                        <span className={styles.recommendationLabel}>x</span>
                        {formatRange(recommendation.xStart, recommendation.xEnd, 'm')}
                      </p>
                      <p className={styles.recommendationLine}>
                        <span className={styles.recommendationLabel}>y</span>
                        {formatRange(recommendation.yStart, recommendation.yEnd, 'm')}
                      </p>
                      <p className={styles.recommendationLine}>
                        <span className={styles.recommendationLabel}>support</span>
                        {formatSupport(recommendation.smoothedGeneratedSupport)} total / {formatSupport(recommendation.smoothedFailedSupport)} failed
                      </p>
                      <p className={styles.recommendationLine}>
                        <span className={styles.recommendationLabel}>teleop dens.</span>
                        {formatSupport(recommendation.teleopDensity)}
                      </p>
                      <p className={styles.recommendationLine}>
                        <span className={styles.recommendationLabel}>confidence</span>
                        {formatPercent(recommendation.confidenceScore)}
                      </p>
                      <p className={styles.recommendationLine}>
                        <span className={styles.recommendationLabel}>region cells</span>
                        {recommendation.cellCount}
                      </p>
                    </article>
                  ))}
                </div>
              )}
            </section>
          )}
        </>
      )}
    </div>
  );
}

export default ObjectDistributionPage;
