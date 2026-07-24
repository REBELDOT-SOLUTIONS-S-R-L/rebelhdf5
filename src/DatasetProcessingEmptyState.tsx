import { Link } from 'react-router-dom';

import styles from './DatasetProcessingPage.module.css';

export function DatasetProcessingEmptyState({
  openedFileCount,
}: {
  openedFileCount: number;
}) {
  return (
    <div className={styles.emptyState}>
      <h2 className={styles.emptyTitle}>Dataset Processing</h2>
      <p className={styles.emptyText}>
        Open one or more HDF5 files in rebelHDF5, then switch to this page to
        cut, merge, append, or convert datasets.
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
