import styles from './DatasetComparisonPage.module.css';

function formatValue(value: unknown): string {
  if (value === null || value === undefined) {
    return '-';
  }

  if (typeof value === 'string') {
    return value;
  }

  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }

  return JSON.stringify(value, null, 2);
}

function isMatrix(value: unknown): value is unknown[][] {
  return Array.isArray(value) && value.length > 0 && value.every(Array.isArray);
}

function indexedEntries<T>(
  values: readonly T[],
  keyPrefix: string,
): { key: string; position: number; value: T }[] {
  return values.map((value, position) => ({
    key: `${keyPrefix}-${position}`,
    position,
    value,
  }));
}

export function DatasetComparisonRawValue({ value }: { value: unknown }) {
  if (isMatrix(value)) {
    const columnCount = Math.max(0, ...value.map((row) => row.length));
    const columns = indexedEntries(
      Array.from({ length: columnCount }, (_, position) => position),
      'column',
    );
    const rows = indexedEntries(value, 'row');

    return (
      <div className={styles.matrixScroller}>
        <table className={styles.matrixTable}>
          <thead>
            <tr>
              <th aria-label="Row index" />
              {columns.map((column) => (
                <th key={column.key}>{column.position}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.key}>
                <th>{row.position}</th>
                {columns.map((column) => (
                  <td key={column.key}>
                    {formatValue(row.value[column.position])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  if (Array.isArray(value)) {
    const entries = indexedEntries(value, 'entry');

    return (
      <div className={styles.matrixScroller}>
        <table className={styles.vectorTable}>
          <thead>
            <tr>
              <th>Index</th>
              <th>Value</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((entry) => (
              <tr key={entry.key}>
                <th>{entry.position}</th>
                <td>{formatValue(entry.value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  return <pre className={styles.rawPre}>{formatValue(value)}</pre>;
}
