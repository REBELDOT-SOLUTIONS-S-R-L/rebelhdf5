import contour from 'plotly.js/lib/contour';
import Plotly from 'plotly.js/lib/core';
import heatmap from 'plotly.js/lib/heatmap';
import scatter from 'plotly.js/lib/scatter';
import scatter3d from 'plotly.js/lib/scatter3d';
import { useEffect, useMemo, useRef, useState } from 'react';
import { type Figure, type PlotParams } from 'react-plotly.js';
import factoryModule from 'react-plotly.js/factory';

Plotly.register([scatter, scatter3d, heatmap, contour]);

const factory =
  typeof factoryModule === 'function'
    ? factoryModule
    : (factoryModule as { default: typeof factoryModule }).default;

const ReactPlot = factory(Plotly as object);

// Sizing strategy (fixes both the progressive-narrowing and scroll-jump bugs):
// - A relative wrapper (width:100%, fixed height) is sized purely by CSS.
// - The plot is rendered ABSOLUTELY positioned inside it, so it is out of layout
//   flow and can never feed its own width back into the flex layout — which is
//   what made the chart's <section> card shrink a little on every remount.
// - The plot width is driven explicitly from the wrapper's measured width via a
//   ResizeObserver, with autosize / responsive / useResizeHandler all off, so
//   Plotly never measures itself and never redraws (and scrolls) on every click.
function Plot(props: PlotParams) {
  const {
    config,
    layout,
    onInitialized,
    onPurge,
    onUpdate,
    style,
    ...plotProps
  } = props;
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState<number | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) {
      return undefined;
    }
    const measuredElement = el;

    function measure() {
      const next = measuredElement.clientWidth;
      if (next > 0) {
        setWidth((current) => (current === next ? current : next));
      }
    }

    measure();

    if (typeof ResizeObserver === 'undefined') {
      return undefined;
    }

    const observer = new ResizeObserver(measure);
    observer.observe(measuredElement);

    return () => {
      observer.disconnect();
    };
  }, []);

  const mergedConfig = useMemo(
    () => ({ responsive: false, ...config }),
    [config],
  );
  const mergedLayout = useMemo(
    () => ({
      ...layout,
      autosize: false,
      ...(width !== null ? { width } : {}),
    }),
    [layout, width],
  );
  const mergedStyle = useMemo(
    () => ({
      // Absolutely positioned inside the relative wrapper so the plot is removed
      // from layout flow and can never feed its own width back into the flex
      // layout (which made the chart's card progressively shrink on remount).
      position: 'absolute' as const,
      top: 0,
      left: 0,
      width: width ?? '100%',
      height: layout.height,
      ...style,
    }),
    [layout.height, style, width],
  );

  return (
    <div
      ref={containerRef}
      style={{
        position: 'relative',
        width: '100%',
        minWidth: 0,
        height: layout.height,
      }}
    >
      {width !== null && (
        <ReactPlot
          {...plotProps}
          config={mergedConfig}
          layout={mergedLayout}
          onInitialized={(
            figure: Readonly<Figure>,
            graphDiv: Readonly<HTMLElement>,
          ) => {
            onInitialized?.(figure, graphDiv);
          }}
          onPurge={(
            figure: Readonly<Figure>,
            graphDiv: Readonly<HTMLElement>,
          ) => {
            onPurge?.(figure, graphDiv);
          }}
          onUpdate={(
            figure: Readonly<Figure>,
            graphDiv: Readonly<HTMLElement>,
          ) => {
            onUpdate?.(figure, graphDiv);
          }}
          style={mergedStyle}
          useResizeHandler={false}
        />
      )}
    </div>
  );
}

export default Plot;
