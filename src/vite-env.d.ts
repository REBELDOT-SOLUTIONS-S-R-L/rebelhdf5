/// <reference types="vite/client" />

declare const __PYTHON_BACKEND_PORT__: number;

// HDF5 compression plugins
declare module '*.so' {
  const src: string;
  export default src;
}

declare module 'plotly.js-dist-min' {
  import type { PlotlyStatic } from 'plotly.js';

  const Plotly: PlotlyStatic;
  export default Plotly;
}

// File System Access API (not yet in all TypeScript lib definitions)
interface FilePickerAcceptType {
  description?: string;
  accept: Record<string, string[]>;
}

interface SaveFilePickerOptions {
  suggestedName?: string;
  types?: FilePickerAcceptType[];
}

// eslint-disable-next-line no-var -- required for global augmentation
declare var showSaveFilePicker:
  | ((options?: SaveFilePickerOptions) => Promise<FileSystemFileHandle>)
  | undefined;
