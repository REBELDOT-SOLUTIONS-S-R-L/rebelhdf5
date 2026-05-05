import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { Figure, PlotParams } from 'react-plotly.js';
import Plotly from 'plotly.js/lib/core';
import contour from 'plotly.js/lib/contour';
import heatmap from 'plotly.js/lib/heatmap';
import scatter from 'plotly.js/lib/scatter';
import scatter3d from 'plotly.js/lib/scatter3d';
import factoryModule from 'react-plotly.js/factory';

Plotly.register([scatter, scatter3d, heatmap, contour]);

const factory = typeof factoryModule === 'function'
  ? factoryModule
  : (factoryModule as { default: typeof factoryModule }).default;

const ReactPlot = factory(Plotly as object);

function Plot(props: PlotParams) {
  const {
    config,
    layout,
    onInitialized,
    onPurge,
    onUpdate,
    style,
    useResizeHandler,
    ...plotProps
  } = props;
  const [graphDiv, setGraphDiv] = useState<HTMLElement | null>(null);
  const resizeFrameRef = useRef<number | null>(null);

  const mergedConfig = useMemo(
    () => ({ responsive: true, ...config }),
    [config],
  );
  const mergedStyle = useMemo(
    () => ({
      width: '100%',
      minWidth: 0,
      height: layout?.height,
      ...style,
    }),
    [layout?.height, style],
  );

  const scheduleResize = useCallback(() => {
    if (typeof window === 'undefined') {
      return;
    }

    if (resizeFrameRef.current !== null) {
      window.cancelAnimationFrame(resizeFrameRef.current);
    }

    resizeFrameRef.current = window.requestAnimationFrame(() => {
      resizeFrameRef.current = null;
      if (graphDiv?.isConnected) {
        Plotly.Plots.resize(graphDiv);
      }
    });
  }, [graphDiv]);

  const rememberGraphDiv = useCallback((nextGraphDiv: Readonly<HTMLElement>) => {
    setGraphDiv((current) => (
      current === nextGraphDiv ? current : nextGraphDiv as HTMLElement
    ));
  }, []);

  useEffect(() => {
    scheduleResize();
  });

  useEffect(() => {
    if (!graphDiv || typeof ResizeObserver === 'undefined') {
      return undefined;
    }

    const resizeTarget = graphDiv.parentElement ?? graphDiv;
    const observer = new ResizeObserver(() => {
      scheduleResize();
    });

    observer.observe(resizeTarget);
    scheduleResize();

    return () => {
      observer.disconnect();
    };
  }, [graphDiv, scheduleResize]);

  useEffect(() => {
    return () => {
      if (resizeFrameRef.current !== null) {
        window.cancelAnimationFrame(resizeFrameRef.current);
      }
    };
  }, []);

  return (
    <ReactPlot
      {...plotProps}
      config={mergedConfig}
      layout={layout}
      onInitialized={(figure: Readonly<Figure>, nextGraphDiv: Readonly<HTMLElement>) => {
        rememberGraphDiv(nextGraphDiv);
        onInitialized?.(figure, nextGraphDiv);
      }}
      onPurge={(figure: Readonly<Figure>, nextGraphDiv: Readonly<HTMLElement>) => {
        onPurge?.(figure, nextGraphDiv);
        setGraphDiv(null);
      }}
      onUpdate={(figure: Readonly<Figure>, nextGraphDiv: Readonly<HTMLElement>) => {
        rememberGraphDiv(nextGraphDiv);
        onUpdate?.(figure, nextGraphDiv);
      }}
      style={mergedStyle}
      useResizeHandler={useResizeHandler ?? true}
    />
  );
}

export default Plot;
