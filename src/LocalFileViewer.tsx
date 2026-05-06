import { H5WasmLocalFileProvider } from '@h5web/h5wasm';
import { ErrorBoundary } from 'react-error-boundary';

import FileErrorFallback from './FileErrorFallback';
import H5WebApp from './H5WebApp';
import { getPlugin } from './plugin-utils';
import { type LocalFile } from './stores';
import styles from './ViewerTheme.module.css';

interface Props {
  file: LocalFile;
}

function LocalFileViewer(props: Props) {
  const { file } = props;
  const { resolvedUrl, file: rawFile } = file;

  return (
    <H5WasmLocalFileProvider file={rawFile} getPlugin={getPlugin}>
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
    </H5WasmLocalFileProvider>
  );
}

export default LocalFileViewer;
