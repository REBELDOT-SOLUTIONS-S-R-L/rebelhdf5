import { lazy, Suspense } from 'react';
import { ErrorBoundary } from 'react-error-boundary';
import { Route, Routes, useLocation } from 'react-router-dom';

import Dropzone from './Dropzone';
import Layout from './Layout';
import Loader from './Loader';
import RouteErrorFallback from './RouteErrorFallback';

const ObjectDistributionPage = lazy(
  async () => import('./ObjectDistributionPage'),
);
const DatasetAttributesPage = lazy(
  async () => import('./DatasetAttributesPage'),
);
const DatabricksPage = lazy(async () => import('./DatabricksPage'));
const DatasetComparisonPage = lazy(
  async () => import('./DatasetComparisonPage'),
);
const DatasetProcessingPage = lazy(
  async () => import('./DatasetProcessingPage'),
);
const PoseTracePage = lazy(async () => import('./PoseTracePage'));
const ServicesPage = lazy(async () => import('./ServicesPage'));
const VideoConverterPage = lazy(async () => import('./VideoConverterPage'));
const ViewPage = lazy(async () => import('./ViewPage'));

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
              <Route
                path="/dataset-processing"
                element={<DatasetProcessingPage />}
              />
              <Route
                path="/dataset-comparison"
                element={<DatasetComparisonPage />}
              />
              <Route
                path="/dataset-attributes"
                element={<DatasetAttributesPage />}
              />
              <Route
                path="/object-distribution"
                element={<ObjectDistributionPage />}
              />
              <Route path="/databricks" element={<DatabricksPage />} />
            </Routes>
          </Suspense>
        </ErrorBoundary>
      </Layout>
    </Dropzone>
  );
}

export default App;
