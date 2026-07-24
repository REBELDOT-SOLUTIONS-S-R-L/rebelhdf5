import styles from './DatasetComparisonPage.module.css';
import { DatasetComparisonRawValue } from './DatasetComparisonRawValue';
import { type DatasetComparisonValue } from './pose-trace/types';

function formatShape(shape: readonly number[] | null): string {
  if (!shape) {
    return '-';
  }

  return shape.length === 0 ? 'scalar' : shape.join(' x ');
}

interface DatasetComparisonValuePanelProps {
  title: string;
  result: DatasetComparisonValue | undefined;
}

export function DatasetComparisonValuePanel({
  title,
  result,
}: DatasetComparisonValuePanelProps) {
  return (
    <div className={styles.valuePanel}>
      <div className={styles.valuePanelHeader}>
        <h4 className={styles.valuePanelTitle}>{title}</h4>
      </div>
      {!result ? (
        <p className={styles.infoText}>No comparison data loaded.</p>
      ) : result.status !== 'ok' ? (
        <p className={styles.errorText}>
          {result.error ?? 'Dataset value could not be read.'}
        </p>
      ) : (
        <>
          <div className={styles.valueMetaGrid}>
            <span>
              <strong>Shape:</strong> {formatShape(result.shape)}
            </span>
            <span>
              <strong>Dtype:</strong> {result.dtype ?? '-'}
            </span>
            <span>
              <strong>Selection:</strong> {result.selection ?? 'full value'}
            </span>
          </div>
          <DatasetComparisonRawValue value={result.value} />
        </>
      )}
    </div>
  );
}
