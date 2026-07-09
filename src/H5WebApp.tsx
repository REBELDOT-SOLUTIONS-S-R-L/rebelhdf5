import { App } from '@h5web/app';

const H5WEB_PREFERRED_VIS_KEY = 'h5web:preferredVis';

try {
  globalThis.localStorage?.setItem(H5WEB_PREFERRED_VIS_KEY, 'Matrix');
} catch {
  // Ignore unavailable storage. H5Web will fall back to its own defaults.
}

function H5WebApp() {
  return <App initialPath="/data" propagateErrors />;
}

export default H5WebApp;
