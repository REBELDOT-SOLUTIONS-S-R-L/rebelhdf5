import { startTransition, useEffect, useMemo, useRef, useState } from 'react';

import {
  build3DDataForStep,
  build3DLayout,
  buildEmptyLayout,
  getDefaultHidden3DTraceGroups,
  getJointChartSpecs,
  type PlotSceneCamera,
} from './pose-trace/plotConfig';
import Plot from './pose-trace/PlotlyChart';
import { type DemoRow, type ParsedArticulation } from './pose-trace/types';
import { PoseTraceJointCharts } from './PoseTraceJointCharts';
import styles from './PoseTracePage.module.css';
import { clonePlotSceneCamera } from './PoseTracePage.utils';

interface PlotLegendTraceState {
  legendgroup?: string;
}

interface PlotLegendClickEvent {
  curveNumber?: number;
  data?: PlotLegendTraceState[];
}

interface PoseTraceChartsProps {
  rows: DemoRow[];
  loading: boolean;
  themeKey: string;
  articulation: ParsedArticulation | null;
}

export function PoseTraceCharts({
  rows,
  loading,
  themeKey,
  articulation,
}: PoseTraceChartsProps) {
  const hasData = rows.length > 0;
  const [selectedStepIndex, setSelectedStepIndex] = useState(0);
  const [hiddenTraceGroupsVersion, setHiddenTraceGroupsVersion] = useState(0);
  const sceneCameraRef = useRef<PlotSceneCamera | null>(null);
  const sceneCameraIdentityRef = useRef<string | null>(null);
  const hiddenTraceGroupsRef = useRef<Set<string>>(new Set());
  const emptyMessage = loading
    ? 'Loading pose-trace data…'
    : 'Select a demo to inspect its end-effector and joint trajectories.';
  const maxStepIndex = Math.max(rows.length - 1, 0);
  const currentStepIndex = Math.min(selectedStepIndex, maxStepIndex);
  const currentStepRow = rows.at(currentStepIndex);
  const currentStepLabel = currentStepRow
    ? currentStepRow.episode_step
    : currentStepIndex;
  const firstRow = rows.at(0);
  const threeDimensionalChartKey = `${firstRow?.dataset_name ?? 'dataset'}-${firstRow?.demo_name ?? 'demo'}-3d-${themeKey}`;
  const defaultHiddenTraceGroups = useMemo(
    () => getDefaultHidden3DTraceGroups(rows),
    [rows],
  );
  const jointSpecs = useMemo(
    () =>
      getJointChartSpecs(
        articulation?.segmentation ?? [],
        articulation?.joints ?? [],
      ),
    [articulation],
  );

  if (sceneCameraIdentityRef.current !== threeDimensionalChartKey) {
    sceneCameraIdentityRef.current = threeDimensionalChartKey;
    sceneCameraRef.current = null;
    hiddenTraceGroupsRef.current = defaultHiddenTraceGroups;
  }

  const threeDimensionalData = useMemo(() => {
    // Both values intentionally invalidate data when the legend state or chart
    // identity changes even though the current state itself lives in refs.
    const dataRevision = `${hiddenTraceGroupsVersion}:${threeDimensionalChartKey}`;
    return hasData && dataRevision
      ? build3DDataForStep(rows, currentStepIndex, hiddenTraceGroupsRef.current)
      : [];
  }, [
    currentStepIndex,
    hasData,
    hiddenTraceGroupsVersion,
    rows,
    threeDimensionalChartKey,
  ]);

  useEffect(() => {
    setSelectedStepIndex(maxStepIndex);
  }, [maxStepIndex, rows]);

  return (
    <div className={styles.chartStack}>
      <section className={styles.chartCard}>
        <Plot
          key={threeDimensionalChartKey}
          data={threeDimensionalData}
          layout={
            hasData
              ? build3DLayout(rows, sceneCameraRef.current)
              : buildEmptyLayout('3D Pose Trace', emptyMessage, true)
          }
          onUpdate={(figure) => {
            const nextCamera = clonePlotSceneCamera(
              figure.layout.scene?.camera,
            );
            if (nextCamera) {
              sceneCameraRef.current = nextCamera;
            }
          }}
          onLegendClick={(event) => {
            const clickEvent = event as PlotLegendClickEvent;
            const clickedTrace =
              clickEvent.data?.[clickEvent.curveNumber ?? -1];
            if (typeof clickedTrace?.legendgroup !== 'string') {
              return false;
            }

            const nextHiddenTraceGroups = new Set(hiddenTraceGroupsRef.current);
            if (nextHiddenTraceGroups.has(clickedTrace.legendgroup)) {
              nextHiddenTraceGroups.delete(clickedTrace.legendgroup);
            } else {
              nextHiddenTraceGroups.add(clickedTrace.legendgroup);
            }

            hiddenTraceGroupsRef.current = nextHiddenTraceGroups;
            startTransition(() => {
              setHiddenTraceGroupsVersion((version) => version + 1);
            });

            return false;
          }}
          onLegendDoubleClick={() => false}
          useResizeHandler
          style={{ width: '100%' }}
          config={{ responsive: true }}
        />
        {hasData && (
          <div className={styles.stepSliderBlock}>
            <div className={styles.stepSliderHeader}>
              <label
                className={styles.stepSliderLabel}
                htmlFor="pose-trace-step-slider"
              >
                3D step
              </label>
              <span className={styles.stepSliderValue}>
                {currentStepLabel} ({currentStepIndex + 1}/{rows.length})
              </span>
            </div>
            <input
              id="pose-trace-step-slider"
              className={styles.stepSlider}
              type="range"
              aria-label="3D pose trace step"
              min={0}
              max={maxStepIndex}
              step={1}
              value={currentStepIndex}
              onChange={(event) => {
                const nextStepIndex = Number(event.target.value);
                startTransition(() => {
                  setSelectedStepIndex(nextStepIndex);
                });
              }}
              disabled={rows.length <= 1}
            />
          </div>
        )}
      </section>

      {jointSpecs.length === 0 && hasData && (
        <section className={styles.messageCard}>
          <p>
            This dataset has no joint mapping attrs, so no per-joint charts can
            be shown.
          </p>
        </section>
      )}

      <PoseTraceJointCharts
        specs={jointSpecs}
        rows={rows}
        hasData={hasData}
        themeKey={themeKey}
      />
    </div>
  );
}
