import { Suspense, lazy } from 'react';
import { Route, Routes } from 'react-router-dom';

import Dropzone from './Dropzone';
import HelpPage from './HelpPage';
import Layout from './Layout';
const DatabricksPage = lazy(() => import('./DatabricksPage'));
const DatasetProcessingPage = lazy(() => import('./DatasetProcessingPage'));
const PoseTracePage = lazy(() => import('./PoseTracePage'));
import ServicesPage from './ServicesPage';
const VideoConverterPage = lazy(() => import('./VideoConverterPage'));
import ViewPage from './ViewPage';

function App() {
  return (
    <Dropzone>
      <Layout>
        <Routes>
          <Route path="/" element={<ServicesPage />} />
          <Route path="/help" element={<HelpPage />} />
          <Route path="/view" element={<ViewPage />} />
          <Route
            path="/pose-trace"
            element={(
              <Suspense fallback={null}>
                <PoseTracePage />
              </Suspense>
            )}
          />
          <Route
            path="/video-converter"
            element={(
              <Suspense fallback={null}>
                <VideoConverterPage />
              </Suspense>
            )}
          />
          <Route
            path="/dataset-processing"
            element={(
              <Suspense fallback={null}>
                <DatasetProcessingPage />
              </Suspense>
            )}
          />
          <Route
            path="/databricks"
            element={(
              <Suspense fallback={null}>
                <DatabricksPage />
              </Suspense>
            )}
          />
        </Routes>
      </Layout>
    </Dropzone>
  );
}

export default App;
