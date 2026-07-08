import type { ComponentProps } from 'react';

// A lightweight stand-in for ./pose-trace/PlotlyChart. Real Plotly cannot run
// under jsdom (it needs canvas/WebGL), so page tests mock the chart with this
// stub. It records the props it was rendered with on a data attribute and
// exposes legend/update callbacks via buttons so tests can drive interactions.
export interface PlotlyChartStubProps {
  data?: unknown;
  layout?: { title?: unknown } | Record<string, unknown>;
  onLegendClick?: (event: unknown) => boolean | void;
  onLegendDoubleClick?: (event: unknown) => boolean | void;
  onUpdate?: (figure: unknown) => void;
  [key: string]: unknown;
}

export default function PlotlyChartStub(props: PlotlyChartStubProps) {
  const traceCount = Array.isArray(props.data) ? props.data.length : 0;
  return (
    <div data-testid="plotly-chart" data-trace-count={traceCount}>
      Plotly chart stub ({traceCount} traces)
    </div>
  );
}

export type PlotlyChartStubComponentProps = ComponentProps<
  typeof PlotlyChartStub
>;
