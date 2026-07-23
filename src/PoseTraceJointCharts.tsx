import { useEffect, useMemo, useState } from 'react';
import { FiPlus, FiX } from 'react-icons/fi';

import {
  buildCombinedJointChartData,
  buildEmptyLayout,
  buildJointChartData,
  buildJointChartLayout,
  type JointChartSpec,
} from './pose-trace/plotConfig';
import Plot from './pose-trace/PlotlyChart';
import { type DemoRow } from './pose-trace/types';
import styles from './PoseTracePage.module.css';
import { jointSpecKey, jointSpecLabel } from './PoseTracePage.utils';

interface PoseTraceJointChartsProps {
  specs: JointChartSpec[];
  rows: DemoRow[];
  hasData: boolean;
  themeKey: string;
}

export function PoseTraceJointCharts({
  specs,
  rows,
  hasData,
  themeKey,
}: PoseTraceJointChartsProps) {
  const [selectedJointIds, setSelectedJointIds] = useState<string[]>([]);
  const [pendingJointId, setPendingJointId] = useState('');
  const [showAllJoints, setShowAllJoints] = useState(false);
  const [combineJoints, setCombineJoints] = useState(false);
  const datasetName = rows[0]?.dataset_name ?? 'dataset';
  const demoName = rows[0]?.demo_name ?? 'demo';
  const availableJointIds = useMemo(() => specs.map(jointSpecKey), [specs]);

  useEffect(() => {
    setSelectedJointIds((current) => {
      const validCurrent = current.filter((id) =>
        availableJointIds.includes(id),
      );
      if (validCurrent.length > 0) {
        return validCurrent;
      }
      return availableJointIds.slice(0, 1);
    });
    setPendingJointId('');
  }, [availableJointIds]);

  const selectedSpecIds = new Set(selectedJointIds);
  const displayedSpecs = showAllJoints
    ? specs
    : specs.filter((spec) => selectedSpecIds.has(jointSpecKey(spec)));
  const addableSpecs = specs.filter(
    (spec) => !selectedSpecIds.has(jointSpecKey(spec)),
  );
  const fallbackSpec = addableSpecs.at(0) ??
    specs.at(0) ?? {
      segmentName: '',
      jointIndex: 0,
    };
  const pendingJointValue = pendingJointId || jointSpecKey(fallbackSpec);

  if (specs.length === 0) {
    return null;
  }

  return (
    <section className={styles.segmentSection}>
      <div className={styles.jointHeader}>
        <h2 className={styles.segmentTitle}>Joints</h2>
        <div className={styles.jointToggles}>
          <label className={styles.checkboxLabel}>
            <input
              type="checkbox"
              aria-label="Show all joints"
              checked={showAllJoints}
              onChange={(event) => {
                setShowAllJoints(event.target.checked);
              }}
            />
            Show all
          </label>
          <label className={styles.checkboxLabel}>
            <input
              type="checkbox"
              aria-label="Combine joint charts"
              checked={combineJoints}
              onChange={(event) => {
                setCombineJoints(event.target.checked);
              }}
            />
            Combine
          </label>
        </div>
      </div>

      <div className={styles.jointControls}>
        <select
          className={styles.select}
          aria-label="Joint to add"
          value={pendingJointValue}
          onChange={(event) => {
            setPendingJointId(event.target.value);
          }}
          disabled={showAllJoints || addableSpecs.length === 0}
        >
          {addableSpecs.length === 0 && (
            <option value="">All joints selected</option>
          )}
          {addableSpecs.map((spec) => (
            <option key={jointSpecKey(spec)} value={jointSpecKey(spec)}>
              {jointSpecLabel(spec)}
            </option>
          ))}
        </select>
        <button
          className={styles.iconButton}
          type="button"
          title="Add joint"
          aria-label="Add joint"
          disabled={showAllJoints || addableSpecs.length === 0}
          onClick={() => {
            const nextId = pendingJointValue;
            if (!nextId) {
              return;
            }
            setSelectedJointIds((current) =>
              current.includes(nextId) ? current : [...current, nextId],
            );
            setPendingJointId('');
          }}
        >
          <FiPlus aria-hidden />
        </button>
      </div>

      {!showAllJoints && selectedJointIds.length > 0 && (
        <div className={styles.jointChips}>
          {displayedSpecs.map((spec) => (
            <span key={jointSpecKey(spec)} className={styles.jointChip}>
              {jointSpecLabel(spec)}
              <button
                className={styles.chipRemoveButton}
                type="button"
                title="Remove joint"
                aria-label={`Remove ${jointSpecLabel(spec)}`}
                onClick={() => {
                  const id = jointSpecKey(spec);
                  setSelectedJointIds((current) =>
                    current.filter((entry) => entry !== id),
                  );
                }}
              >
                <FiX aria-hidden />
              </button>
            </span>
          ))}
        </div>
      )}

      {combineJoints ? (
        <div className={styles.chartCard}>
          <Plot
            key={`${datasetName}-${demoName}-combined-joints-${themeKey}-${displayedSpecs.map(jointSpecKey).join('|')}`}
            data={
              hasData ? buildCombinedJointChartData(rows, displayedSpecs) : []
            }
            layout={
              hasData
                ? buildJointChartLayout(displayedSpecs)
                : buildEmptyLayout('Selected joints', 'No data')
            }
            useResizeHandler
            style={{ width: '100%' }}
            config={{ responsive: true }}
          />
        </div>
      ) : (
        <div className={styles.jointGrid}>
          {displayedSpecs.map((spec) => (
            <div key={jointSpecKey(spec)} className={styles.chartCard}>
              <Plot
                key={`${datasetName}-${demoName}-${jointSpecKey(spec)}-${themeKey}`}
                data={hasData ? buildJointChartData(rows, spec) : []}
                layout={
                  hasData
                    ? buildJointChartLayout(spec)
                    : buildEmptyLayout(jointSpecLabel(spec), 'No data')
                }
                useResizeHandler
                style={{ width: '100%' }}
                config={{ responsive: true }}
              />
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
