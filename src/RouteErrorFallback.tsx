import { type FallbackProps } from 'react-error-boundary';

import { formatUnknownError } from './error-utils';
import styles from './ErrorFallback.module.css';

function RouteErrorFallback(props: FallbackProps) {
  const { error, resetErrorBoundary } = props;

  return (
    <div className={styles.root} data-error-fallback>
      <div className={styles.error}>
        <p>Page could not be loaded.</p>
        <p>{formatUnknownError(error)}</p>
      </div>
      <button className={styles.btn} type="button" onClick={resetErrorBoundary}>
        Retry
      </button>
    </div>
  );
}

export default RouteErrorFallback;
