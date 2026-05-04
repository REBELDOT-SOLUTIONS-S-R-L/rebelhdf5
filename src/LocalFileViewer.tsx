import { App } from '@h5web/app';
import { H5WasmLocalFileProvider } from '@h5web/h5wasm';
import { ErrorBoundary } from 'react-error-boundary';

import FileErrorFallback from './FileErrorFallback';
import { getPlugin } from './plugin-utils';
import { type LocalFile } from './stores';
import { buildMailto, FEEDBACK_MESSAGE } from './utils';
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
          <App
            key={resolvedUrl}
            initialPath="/data"
            propagateErrors
            getFeedbackURL={({ entityPath }) => {
              return buildMailto('Feedback', FEEDBACK_MESSAGE, file, entityPath);
            }}
          />
        </ErrorBoundary>
      </div>
    </H5WasmLocalFileProvider>
  );
}

export default LocalFileViewer;
