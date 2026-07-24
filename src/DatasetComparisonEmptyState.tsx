import { Link } from 'react-router-dom';

import styles from './DatasetComparisonPage.module.css';

interface DatasetComparisonEmptyStateProps {
  openedFileCount: number;
}

export function DatasetComparisonEmptyState({
  openedFileCount,
}: DatasetComparisonEmptyStateProps) {
  return (
    <div className={styles.emptyState}>
      <h2 className={styles.emptyTitle}>Dataset Comparison</h2>
      <p className={styles.emptyText}>
        Open an HDF5 file in rebelHDF5, then compare demo keys side by side.
      </p>
      <div className={styles.emptyActions}>
        <Link className={styles.openBtn} to="/">
          Open HDF5
        </Link>
        {openedFileCount > 0 && (
          <span>
            {openedFileCount} opened file{openedFileCount === 1 ? '' : 's'}{' '}
            available in the sidebar.
          </span>
        )}
      </div>
    </div>
  );
}
