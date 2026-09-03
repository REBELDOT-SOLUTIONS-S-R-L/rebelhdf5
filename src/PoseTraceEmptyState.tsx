import { Link } from 'react-router-dom';

import styles from './PoseTracePage.module.css';

export function PoseTraceEmptyState({
  openedFileCount,
}: {
  openedFileCount: number;
}) {
  return (
    <div className={styles.emptyState}>
      <h2 className={styles.emptyTitle}>Pose Trace</h2>
      <p className={styles.emptyText}>
        Open an HDF5 file in rebelHDF5, then switch to this page to plot the
        pose-trace data.
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
