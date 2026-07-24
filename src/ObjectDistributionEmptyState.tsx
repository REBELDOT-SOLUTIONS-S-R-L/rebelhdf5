import { Link } from 'react-router-dom';

import styles from './ObjectDistributionPage.module.css';

export function ObjectDistributionEmptyState({
  openedFileCount,
}: {
  openedFileCount: number;
}) {
  return (
    <div className={styles.emptyState}>
      <h2 className={styles.emptyTitle}>Object Distribution</h2>
      <p className={styles.emptyText}>
        Open HDF5 files in rebelHDF5, then use this page to compare generated
        datasets, inspect source-demo links, and analyze failure coverage across
        the object reset space.
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
