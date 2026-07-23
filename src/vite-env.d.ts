/// <reference types="vite/client" />

declare const __PYTHON_BACKEND_PORT__: number;

interface RebelHdf5DesktopRuntime {
  backendPort?: number;
  chooseDirectory?: (defaultPath?: string) => Promise<string | undefined>;
  getPathForFile?: (file: File) => string | undefined;
  readFile?: (path: string) => Promise<ArrayBuffer>;
}

interface Window {
  rebelHdf5Desktop?: RebelHdf5DesktopRuntime;
}

// eslint-disable-next-line no-var -- required for global augmentation
declare var rebelHdf5Desktop: RebelHdf5DesktopRuntime | undefined;

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

type WellKnownDirectory =
  | 'desktop'
  | 'documents'
  | 'downloads'
  | 'music'
  | 'pictures'
  | 'videos';

interface OpenFilePickerOptions {
  excludeAcceptAllOption?: boolean;
  id?: string;
  multiple?: boolean;
  startIn?: WellKnownDirectory | FileSystemHandle;
  types?: FilePickerAcceptType[];
}

// eslint-disable-next-line no-var -- required for global augmentation
declare var showOpenFilePicker:
  | ((options?: OpenFilePickerOptions) => Promise<FileSystemFileHandle[]>)
  | undefined;

// eslint-disable-next-line no-var -- required for global augmentation
declare var showSaveFilePicker:
  | ((options?: SaveFilePickerOptions) => Promise<FileSystemFileHandle>)
  | undefined;
