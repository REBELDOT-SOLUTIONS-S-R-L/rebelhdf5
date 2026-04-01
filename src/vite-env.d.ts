/// <reference types="vite/client" />

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
