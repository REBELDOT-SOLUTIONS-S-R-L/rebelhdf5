import { type ComponentProps } from 'react';

// A lightweight stand-in for ./pose-trace/PlotlyChart. Real Plotly cannot run
// under jsdom (it needs canvas/WebGL), so page tests mock the chart with this
// stub. It records the props it was rendered with on a data attribute and
// exposes legend/update callbacks via buttons so tests can drive interactions.
export interface PlotlyChartStubProps {
  [key: string]: unknown;
  data?: unknown;
  layout?: { title?: unknown } | Record<string, unknown>;
  onLegendClick?: (event: unknown) => boolean | undefined;
  onLegendDoubleClick?: (event: unknown) => boolean | undefined;
  onUpdate?: (figure: unknown) => void;
}

export default function PlotlyChartStub(props: PlotlyChartStubProps) {
  const { data, layout, onLegendClick, onLegendDoubleClick, onUpdate } = props;
  const traceCount = Array.isArray(data) ? data.length : 0;
  return (
    <div
      data-testid="plotly-chart"
      data-trace-count={traceCount}
      data-has-layout={layout !== undefined}
      data-has-legend-click={onLegendClick !== undefined}
      data-has-legend-double-click={onLegendDoubleClick !== undefined}
      data-has-update={onUpdate !== undefined}
    >
      Plotly chart stub ({traceCount} traces)
    </div>
  );
}

export type PlotlyChartStubComponentProps = ComponentProps<
  typeof PlotlyChartStub
>;
