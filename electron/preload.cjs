'use strict';

const { contextBridge, webUtils } = require('electron');

function readNumberArg(name) {
  const prefix = `--${name}=`;
  const raw = process.argv.find((arg) => arg.startsWith(prefix))?.slice(prefix.length);
  const parsed = raw ? Number(raw) : NaN;
  return Number.isInteger(parsed) && parsed > 0 ? parsed : undefined;
}

contextBridge.exposeInMainWorld('rebelHdf5Desktop', {
  backendPort: readNumberArg('backend-port'),
  getPathForFile(file) {
    if (!file || typeof webUtils.getPathForFile !== 'function') {
      return undefined;
    }

    return webUtils.getPathForFile(file) || undefined;
  },
});
