import { H5WasmBufferProvider } from '@h5web/h5wasm';
import { ErrorBoundary } from 'react-error-boundary';
import { suspend } from 'suspend-react';

import { fetchBuffer } from './fetch-utils';
import FileErrorFallback from './FileErrorFallback';
import H5WebApp from './H5WebApp';
import { getPlugin } from './plugin-utils';
import { type RemoteFile } from './stores';
import styles from './ViewerTheme.module.css';

export const FETCH_BUFFER_KEY = Symbol('fetchBuffer');

interface Props {
  file: RemoteFile;
}

function RemoteFileViewer(props: Props) {
  const { file } = props;
  const { name, resolvedUrl } = file;

  const buffer = suspend(fetchBuffer, [resolvedUrl, FETCH_BUFFER_KEY]);

  return (
    <H5WasmBufferProvider filename={name} buffer={buffer} getPlugin={getPlugin}>
      <div className={styles.viewerTheme}>
        <ErrorBoundary
          fallbackRender={(fallbackProps) => (
            <FileErrorFallback file={file} {...fallbackProps} />
          )}
          resetKeys={[file]}
        >
          <H5WebApp key={resolvedUrl} />
        </ErrorBoundary>
      </div>
    </H5WasmBufferProvider>
  );
}

export default RemoteFileViewer;
