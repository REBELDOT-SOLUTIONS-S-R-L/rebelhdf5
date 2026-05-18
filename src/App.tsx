import { lazy, Suspense } from 'react';
import { type FallbackProps, ErrorBoundary } from 'react-error-boundary';
import { Route, Routes, useLocation } from 'react-router-dom';

import Dropzone from './Dropzone';
import styles from './ErrorFallback.module.css';
import Layout from './Layout';
import Loader from './Loader';

const ClothDistributionPage = lazy(async () => import('./ClothDistributionPage'));
const DatasetAttributesPage = lazy(async () => import('./DatasetAttributesPage'));
const DatabricksPage = lazy(async () => import('./DatabricksPage'));
const DatasetProcessingPage = lazy(async () => import('./DatasetProcessingPage'));
const PoseTracePage = lazy(async () => import('./PoseTracePage'));
const ServicesPage = lazy(async () => import('./ServicesPage'));
const VideoConverterPage = lazy(async () => import('./VideoConverterPage'));
const ViewPage = lazy(async () => import('./ViewPage'));

function RouteErrorFallback(props: FallbackProps) {
  const { error, resetErrorBoundary } = props;
  const msg = error instanceof Error ? error.message : String(error);

  return (
    <div className={styles.root} data-error-fallback>
      <div className={styles.error}>
        <p>Page could not be loaded.</p>
        <p>{msg}</p>
      </div>
      <button className={styles.btn} type="button" onClick={resetErrorBoundary}>
        Retry
      </button>
    </div>
  );
}

function App() {
  const location = useLocation();

  return (
    <Dropzone>
      <Layout>
        <ErrorBoundary
          FallbackComponent={RouteErrorFallback}
          resetKeys={[location.pathname, location.search]}
        >
          <Suspense fallback={<Loader message="Loading page..." />}>
            <Routes>
              <Route path="/" element={<ServicesPage />} />
              <Route path="/view" element={<ViewPage />} />
              <Route path="/pose-trace" element={<PoseTracePage />} />
              <Route path="/video-converter" element={<VideoConverterPage />} />
              <Route path="/dataset-processing" element={<DatasetProcessingPage />} />
              <Route path="/dataset-attributes" element={<DatasetAttributesPage />} />
              <Route path="/cloth-distribution" element={<ClothDistributionPage />} />
              <Route path="/databricks" element={<DatabricksPage />} />
            </Routes>
          </Suspense>
        </ErrorBoundary>
      </Layout>
    </Dropzone>
  );
}

export default App;
