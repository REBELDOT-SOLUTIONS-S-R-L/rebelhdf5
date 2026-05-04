import Plotly from 'plotly.js-dist-min';
// eslint-disable-next-line @typescript-eslint/no-require-imports
import factoryModule from 'react-plotly.js/factory';

const factory = typeof factoryModule === 'function'
  ? factoryModule
  : (factoryModule as { default: typeof factoryModule }).default;

const Plot = factory(Plotly as object);
export default Plot;
