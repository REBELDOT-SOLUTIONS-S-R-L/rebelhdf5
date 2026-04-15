import type Plotly from 'plotly.js';
import type { Data, Layout } from 'plotly.js';

import type { FailurePlane, FailureSlice } from './clothAnalysis';
import {
  TRACE_3D_SPECS,
  TRACE_EEF_KEYPOINT_COLUMNS,
  TRACE_TERM_COLUMNS,
  TRACE_Z_COLUMNS,
  humanizeColumnName,
} from './schema';
import type {
  ClothDistributionAnchor,
  ClothDistributionPoint,
  ClothDistributionResult,
  ClothDistributionSourceDetail,
  DemoRow,
} from './types';

export type TraceSide = 'left' | 'right';
type LegendId = 'legend' | 'legend2' | 'legend3';
type SceneVector = { x?: number; y?: number; z?: number };

export interface PlotSceneCamera {
  center?: SceneVector;
  eye?: SceneVector;
  up?: SceneVector;
  projection?: { type?: string };
}

type PlotTraceVisibility = true | false | 'legendonly';

interface TracePoint3D {
  rowIndex: number;
  x: number;
  y: number;
  z: number;
}

const FONT_FAMILY = 'Roboto, sans-serif';
const PLOTLY_WHITE_TEMPLATE = 'plotly_white' as unknown as Layout['template'];
const FUTURE_TRACE_OPACITY = 0.22;
const FUTURE_MARKER_OPACITY = 0.35;
const DEFAULT_3D_CAMERA_EYE = { x: 0.0, y: -1.8, z: 0.5 };
const DEFAULT_3D_CAMERA_UP = { x: 0, y: 0, z: 1 };
const FAILURE_COLORSCALE: Plotly.ColorScale = [
  [0, '#2ca02c'],
  [0.5, '#ffd000'],
  [1, '#d62728'],
];

const SUBPLOT_DOMAINS: [number, number][] = [
  [0.74, 1.0],
  [0.4, 0.62],
  [0.06, 0.28],
];

const SUBPLOT_TITLES = [
  'EEF To Keypoint Distances',
  'Garment Fold-Term Distances',
  'EEF And Garment Keypoint Heights',
];

interface PlotTheme {
  paperBg: string;
  plotBg: string;
  gridColor: string;
  textColor: string;
  mutedText: string;
  legendBg: string;
  legendBorder: string;
}

function cssColor(name: string, fallback: string): string {
  if (typeof window === 'undefined') {
    return fallback;
  }

  const value = window.getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

function getPlotTheme(): PlotTheme {
  return {
    paperBg: cssColor('--color-plot-paper', '#ffffff'),
    plotBg: cssColor('--color-plot-surface', '#f7f7fa'),
    gridColor: cssColor('--color-plot-grid', 'rgba(90, 90, 102, 0.14)'),
    textColor: cssColor('--color-text', '#5a5a66'),
    mutedText: cssColor('--color-text-muted', '#7b7b87'),
    legendBg: cssColor('--color-plot-legend-bg', 'rgba(255, 255, 255, 0.92)'),
    legendBorder: cssColor('--color-plot-legend-border', 'rgba(90, 90, 102, 0.14)'),
  };
}

function xAxis(rows: DemoRow[]): number[] {
  if (rows.length > 0 && rows.every((row) => row.episode_step != null)) {
    return rows.map((row) => row.episode_step);
  }

  return rows.map((_, index) => index);
}

function series(rows: DemoRow[], key: string): Array<number | null> {
  return rows.map((row) => {
    const value = row[key];
    return typeof value === 'number' ? value : null;
  });
}

function hasValues(values: Array<number | null>): boolean {
  return values.some((value) => value != null);
}

function matchesSide(column: string, side: TraceSide): boolean {
  return column.includes(`_${side}_`);
}

function sideLabel(side: TraceSide): string {
  return side === 'left' ? 'Left EEF' : 'Right EEF';
}

function build2DLegendLabel(column: string, side: TraceSide): string {
  let label = humanizeColumnName(column);

  if (side === 'left') {
    label = label.replace('left arm', 'eef');
    label = label.replaceAll('garment left ', '');
    label = label.replaceAll('left ', '');
  } else {
    label = label.replace('right arm', 'eef');
    label = label.replaceAll('garment right ', '');
    label = label.replaceAll('right ', '');
  }

  label = label.replaceAll('garment ', '');
  return label.trim();
}

function buildLegendLayout(theme: PlotTheme, y: number): Partial<Layout['legend']> {
  return {
    orientation: 'h',
    yanchor: 'top',
    y,
    xanchor: 'center',
    x: 0.5,
    bgcolor: theme.legendBg,
    bordercolor: theme.legendBorder,
    borderwidth: 1,
    font: { color: theme.textColor, family: FONT_FAMILY, size: 10 },
  };
}

function build3DPoints(rows: DemoRow[], prefix: string): TracePoint3D[] {
  const points: TracePoint3D[] = [];

  rows.forEach((row, rowIndex) => {
    const x = row[`${prefix}_x`];
    const y = row[`${prefix}_y`];
    const z = row[`${prefix}_z`];

    if (typeof x !== 'number' || typeof y !== 'number' || typeof z !== 'number') {
      return;
    }

    points.push({ rowIndex, x, y, z });
  });

  return points;
}

function split3DPoints(points: TracePoint3D[], stepIndex: number): {
  past: TracePoint3D[];
  future: TracePoint3D[];
  current: TracePoint3D | undefined;
} {
  const past = points.filter((point) => point.rowIndex <= stepIndex);
  const current = [...points].reverse().find((point) => point.rowIndex <= stepIndex);
  const futureBase = points.filter((point) => point.rowIndex >= stepIndex);
  const future = current && futureBase[0]?.rowIndex !== current.rowIndex
    ? [current, ...futureBase]
    : futureBase;

  return { past, future, current };
}

function coordinates(points: TracePoint3D[]): { x: number[]; y: number[]; z: number[] } {
  return {
    x: points.map((point) => point.x),
    y: points.map((point) => point.y),
    z: points.map((point) => point.z),
  };
}

function buildTraceVisibility(
  hiddenLegendGroups: ReadonlySet<string> | undefined,
  legendGroup: string,
  showInLegend: boolean,
): PlotTraceVisibility {
  if (!hiddenLegendGroups?.has(legendGroup)) {
    return true;
  }

  return showInLegend ? 'legendonly' : false;
}

function formatClothScalar(value: number | null): string {
  return value == null ? '-' : value.toFixed(4);
}

function buildClothHoverTemplate(category: string): string {
  return [
    '<b>%{customdata[0]}</b>',
    'episode: %{customdata[1]}',
    `category: ${category}`,
    'anchor x: %{x:.4f} m',
    'anchor y: %{y:.4f} m',
    'initial x: %{customdata[2]}',
    'initial y: %{customdata[3]}',
    'initial rx: %{customdata[4]}',
    'initial ry: %{customdata[5]}',
    'num samples: %{customdata[6]}',
    'source left: %{customdata[7]}',
    'source right: %{customdata[8]}',
    '<extra></extra>',
  ].join('<br>');
}

function buildClothScatterTrace(
  points: ClothDistributionPoint[],
  name: 'Success' | 'Failed' | 'Teleop',
  color: string,
  opacity: number,
  hoverEnabled: boolean,
): Data | null {
  if (points.length === 0) {
    return null;
  }

  return {
    type: 'scatter',
    x: points.map((point) => point.x),
    y: points.map((point) => point.y),
    mode: 'markers',
    name,
    marker: {
      color,
      size: 11,
      opacity,
      line: { color: 'rgba(255,255,255,0.7)', width: 0.6 },
    },
    customdata: points.map((point) => [
      point.datasetName,
      point.demoName,
      formatClothScalar(point.initialX),
      formatClothScalar(point.initialY),
      formatClothScalar(point.initialRx),
      formatClothScalar(point.initialRy),
      point.numSamples == null ? '-' : String(point.numSamples),
      point.sourceLeft,
      point.sourceRight,
    ]),
    hovertemplate: hoverEnabled ? buildClothHoverTemplate(name.toLowerCase()) : '<extra></extra>',
    hoverinfo: hoverEnabled ? 'all' : 'skip',
  };
}

function buildClothSelectedEpisodeHover(point: ClothDistributionPoint): string {
  return [
    `<b>${point.datasetName}</b>`,
    `episode: ${point.demoName}`,
    `category: ${point.category}`,
    `anchor x: ${point.x.toFixed(4)} m`,
    `anchor y: ${point.y.toFixed(4)} m`,
    `initial x: ${formatClothScalar(point.initialX)}`,
    `initial y: ${formatClothScalar(point.initialY)}`,
    `initial rx: ${formatClothScalar(point.initialRx)}`,
    `initial ry: ${formatClothScalar(point.initialRy)}`,
    `num samples: ${point.numSamples == null ? '-' : String(point.numSamples)}`,
    `source left: ${point.sourceLeft}`,
    `source right: ${point.sourceRight}`,
  ].join('<br>');
}

function buildClothSourceHoverText(detail: ClothDistributionSourceDetail): string {
  return [
    `<b>${detail.datasetName}</b>`,
    `episode: ${detail.demoName}`,
    detail.hoverLabel,
    `anchor x: ${detail.x.toFixed(4)} m`,
    `anchor y: ${detail.y.toFixed(4)} m`,
  ].join('<br>');
}

function buildClothSourceOverlay(
  details: ClothDistributionSourceDetail[],
  selectedPoint: ClothDistributionPoint,
  markerColor: string,
  textColor: string,
  lineColor: string,
  markerName: string,
  lineName: string,
): Data[] {
  if (details.length === 0) {
    return [];
  }

  return [
    {
      type: 'scatter',
      x: details.flatMap((detail) => [selectedPoint.x, detail.x, null]),
      y: details.flatMap((detail) => [selectedPoint.y, detail.y, null]),
      mode: 'lines',
      name: lineName,
      showlegend: false,
      hoverinfo: 'skip',
      line: { color: lineColor, width: 2.4 },
    },
    {
      type: 'scatter',
      x: details.map((detail) => detail.x),
      y: details.map((detail) => detail.y),
      mode: 'text+markers',
      name: markerName,
      showlegend: false,
      text: details.map((detail) => detail.textLabel),
      textposition: 'top center',
      textfont: { color: textColor, size: 12, family: FONT_FAMILY },
      hovertext: details.map(buildClothSourceHoverText),
      hovertemplate: '%{hovertext}<extra></extra>',
      marker: {
        size: 16,
        color: markerColor,
        opacity: 0.98,
        line: { color: 'rgba(0,0,0,0.7)', width: 1.5 },
      },
    },
  ];
}

export function buildClothDistributionData(
  result: ClothDistributionResult | null,
  selectedPoint: ClothDistributionPoint | null,
): Data[] {
  if (!result) {
    return [];
  }

  const baseOpacity = selectedPoint ? 0.14 : 0.82;
  const baseHoverEnabled = !selectedPoint;
  const traces: Data[] = [];

  const successTrace = buildClothScatterTrace(
    result.successPoints,
    'Success',
    '#2ca02c',
    baseOpacity,
    baseHoverEnabled,
  );
  const failedTrace = buildClothScatterTrace(
    result.failedPoints,
    'Failed',
    '#d62728',
    baseOpacity,
    baseHoverEnabled,
  );
  const teleopTrace = buildClothScatterTrace(
    result.teleopPoints,
    'Teleop',
    '#1f77b4',
    baseOpacity,
    baseHoverEnabled,
  );

  if (successTrace) traces.push(successTrace);
  if (failedTrace) traces.push(failedTrace);
  if (teleopTrace) traces.push(teleopTrace);

  if (!selectedPoint || selectedPoint.category === 'teleop') {
    return traces;
  }

  traces.push({
    type: 'scatter',
    x: [selectedPoint.x],
    y: [selectedPoint.y],
    mode: 'text+markers',
    name: 'Selected Episode',
    showlegend: false,
    text: [selectedPoint.demoName],
    textposition: 'top center',
    textfont: { color: 'rgba(0, 90, 130, 1)', size: 12, family: FONT_FAMILY },
    hovertext: [buildClothSelectedEpisodeHover(selectedPoint)],
    hovertemplate: '%{hovertext}<extra></extra>',
    marker: {
      size: 16,
      color: 'rgba(0, 190, 255, 0.95)',
      opacity: 0.95,
      line: { color: 'rgba(0,0,0,0.7)', width: 1.5 },
    },
  });

  traces.push(
    ...buildClothSourceOverlay(
      selectedPoint.sourceLeftDetails,
      selectedPoint,
      'rgba(255, 215, 0, 0.98)',
      'rgba(125, 95, 0, 1)',
      'rgba(240, 200, 0, 0.95)',
      'Selected Left Sources',
      'Selected Left Links',
    ),
    ...buildClothSourceOverlay(
      selectedPoint.sourceRightDetails,
      selectedPoint,
      'rgba(255, 140, 0, 0.98)',
      'rgba(145, 70, 0, 1)',
      'rgba(255, 140, 0, 0.95)',
      'Selected Right Sources',
      'Selected Right Links',
    ),
  );

  return traces;
}

export function buildClothDistributionLayout(
  _result: ClothDistributionResult | null,
  anchor: ClothDistributionAnchor,
): Partial<Layout> {
  const theme = getPlotTheme();

  return {
    template: PLOTLY_WHITE_TEMPLATE,
    height: 820,
    paper_bgcolor: theme.paperBg,
    plot_bgcolor: theme.plotBg,
    font: { color: theme.textColor, family: FONT_FAMILY },
    hovermode: 'closest',
    margin: { l: 70, r: 30, t: 40, b: 70 },
    legend: {
      orientation: 'h',
      yanchor: 'bottom',
      y: 1.02,
      xanchor: 'left',
      x: 0,
      bgcolor: theme.legendBg,
      bordercolor: theme.legendBorder,
      borderwidth: 1,
    },
    xaxis: {
      title: { text: `${anchor} x [m]` },
      gridcolor: theme.gridColor,
      zeroline: true,
      zerolinewidth: 1,
      zerolinecolor: theme.gridColor,
      color: theme.textColor,
    },
    yaxis: {
      title: { text: `${anchor} y [m]` },
      gridcolor: theme.gridColor,
      zeroline: true,
      zerolinewidth: 1,
      zerolinecolor: theme.gridColor,
      scaleanchor: 'x',
      scaleratio: 1,
      color: theme.textColor,
    },
  };
}

function buildFailureHeatmapTrace(
  plane: FailurePlane,
  axisRefs?: { x: string; y: string },
  showScale = true,
): Data {
  const trace: Record<string, unknown> = {
    type: 'heatmap',
    x: plane.bins[0]?.map((bin) => bin.xCenter) ?? [],
    y: plane.bins.map((row) => row[0]?.yCenter ?? 0),
    z: plane.bins.map((row) => row.map((bin) => (bin.masked ? null : bin.failureRate))),
    customdata: plane.bins.map((row) => row.map((bin) => [
      bin.xStart,
      bin.xEnd,
      bin.yStart,
      bin.yEnd,
      bin.failedCount,
      bin.successCount,
      bin.totalGeneratedCount,
      bin.teleopCount,
      bin.failureRate,
    ])),
    hoverongaps: false,
    hovertemplate: [
      `${plane.xLabel} range: %{customdata[0]:.4f} to %{customdata[1]:.4f}`,
      `${plane.yLabel} range: %{customdata[2]:.4f} to %{customdata[3]:.4f}`,
      'failure rate: %{customdata[8]:.1%}',
      'failed: %{customdata[4]}',
      'success: %{customdata[5]}',
      'total generated: %{customdata[6]}',
      'teleop: %{customdata[7]}',
      '<extra></extra>',
    ].join('<br>'),
    coloraxis: 'coloraxis',
    showscale: showScale,
    xaxis: axisRefs?.x,
    yaxis: axisRefs?.y,
  };

  return trace as unknown as Data;
}

function buildFailureOverlayTrace(
  plane: FailurePlane,
  axisRefs?: { x: string; y: string },
  showLegend = true,
): Data | null {
  if (plane.overlayPoints.length === 0) {
    return null;
  }

  return {
    type: 'scatter',
    x: plane.overlayPoints.map((point) => point.x),
    y: plane.overlayPoints.map((point) => point.y),
    mode: 'markers',
    name: 'Teleop',
    showlegend: showLegend,
    marker: {
      color: '#1f77b4',
      size: 7,
      opacity: 0.55,
      line: { color: 'rgba(255,255,255,0.6)', width: 0.6 },
    },
    hovertemplate: [
      'Teleop coverage',
      `${plane.xLabel}: %{x:.4f}`,
      `${plane.yLabel}: %{y:.4f}`,
      '<extra></extra>',
    ].join('<br>'),
    xaxis: axisRefs?.x,
    yaxis: axisRefs?.y,
  };
}

function buildFailureAxisBase(theme: PlotTheme): Partial<Layout['xaxis']> {
  return {
    showgrid: true,
    gridcolor: theme.gridColor,
    zeroline: false,
    color: theme.textColor,
  };
}

export function buildFailureMapData(
  plane: FailurePlane,
  showTeleopOverlay: boolean,
): Data[] {
  const traces: Data[] = [buildFailureHeatmapTrace(plane)];
  const overlayTrace = showTeleopOverlay ? buildFailureOverlayTrace(plane) : null;
  if (overlayTrace) {
    traces.push(overlayTrace);
  }

  return traces;
}

export function buildFailureMapLayout(
  plane: FailurePlane,
): Partial<Layout> {
  const theme = getPlotTheme();
  const axisBase = buildFailureAxisBase(theme);

  const layout: Partial<Layout> & Record<string, unknown> = {
    template: PLOTLY_WHITE_TEMPLATE,
    height: 820,
    paper_bgcolor: theme.paperBg,
    plot_bgcolor: theme.plotBg,
    font: { color: theme.textColor, family: FONT_FAMILY },
    hovermode: 'closest',
    margin: { l: 70, r: 80, t: 30, b: 70 },
    legend: {
      orientation: 'h',
      yanchor: 'bottom',
      y: 1.01,
      xanchor: 'right',
      x: 1,
      bgcolor: theme.legendBg,
      bordercolor: theme.legendBorder,
      borderwidth: 1,
    },
    xaxis: {
      ...axisBase,
      title: { text: plane.xLabel },
      range: [plane.xBounds.min, plane.xBounds.max],
    },
    yaxis: {
      ...axisBase,
      title: { text: plane.yLabel },
      range: [plane.yBounds.min, plane.yBounds.max],
      scaleanchor: 'x',
      scaleratio: 1,
    },
  };

  layout.coloraxis = {
    cmin: 0,
    cmax: 1,
    colorscale: FAILURE_COLORSCALE,
    colorbar: {
      title: { text: 'failure rate' },
      tickformat: '.0%',
    },
  };

  return layout;
}

function sliceAxisName(prefix: 'x' | 'y', index: number): string {
  return `${prefix}${index === 0 ? '' : index + 1}`;
}

function sliceDomain(index: number, total: number, start: number, end: number, gap: number): [number, number] {
  const width = (end - start - (gap * (total - 1))) / total;
  const domainStart = start + (index * (width + gap));
  return [domainStart, domainStart + width];
}

function formatSliceAnnotation(slice: FailureSlice): string {
  return [
    `rot x: ${slice.rotXStart.toFixed(2)} to ${slice.rotXEnd.toFixed(2)} deg`,
    `rot y: ${slice.rotYStart.toFixed(2)} to ${slice.rotYEnd.toFixed(2)} deg`,
  ].join('<br>');
}

export function buildFailureSliceData(
  slices: FailureSlice[],
  showTeleopOverlay: boolean,
): Data[] {
  const traces: Data[] = [];

  for (const slice of slices) {
    const axisRefs = {
      x: sliceAxisName('x', (slice.rowIndex * 3) + slice.colIndex),
      y: sliceAxisName('y', (slice.rowIndex * 3) + slice.colIndex),
    };

    traces.push(buildFailureHeatmapTrace(slice.plane, axisRefs, false));

    const overlayTrace = showTeleopOverlay
      ? buildFailureOverlayTrace(slice.plane, axisRefs, slice.rowIndex === 0 && slice.colIndex === 0)
      : null;
    if (overlayTrace) {
      traces.push(overlayTrace);
    }
  }

  return traces;
}

export function buildFailureSliceLayout(
  slices: FailureSlice[],
): Partial<Layout> {
  const theme = getPlotTheme();
  const axisBase = buildFailureAxisBase(theme);
  const layout: Partial<Layout> & Record<string, unknown> = {
    template: PLOTLY_WHITE_TEMPLATE,
    height: 1220,
    paper_bgcolor: theme.paperBg,
    plot_bgcolor: theme.plotBg,
    font: { color: theme.textColor, family: FONT_FAMILY },
    hovermode: 'closest',
    margin: { l: 70, r: 90, t: 40, b: 80 },
    legend: {
      orientation: 'h',
      yanchor: 'bottom',
      y: 1.01,
      xanchor: 'right',
      x: 1,
      bgcolor: theme.legendBg,
      bordercolor: theme.legendBorder,
      borderwidth: 1,
    },
    annotations: [] as NonNullable<Layout['annotations']>,
  };

  layout.coloraxis = {
    cmin: 0,
    cmax: 1,
    colorscale: FAILURE_COLORSCALE,
    colorbar: {
      title: { text: 'failure rate' },
      tickformat: '.0%',
    },
  };

  const xDomains = [
    sliceDomain(0, 3, 0.05, 0.97, 0.04),
    sliceDomain(1, 3, 0.05, 0.97, 0.04),
    sliceDomain(2, 3, 0.05, 0.97, 0.04),
  ];
  const yDomains = [
    sliceDomain(0, 3, 0.06, 0.94, 0.04),
    sliceDomain(1, 3, 0.06, 0.94, 0.04),
    sliceDomain(2, 3, 0.06, 0.94, 0.04),
  ].reverse() as [number, number][];

  for (const slice of slices) {
    const axisIndex = (slice.rowIndex * 3) + slice.colIndex;
    const xAxisName = sliceAxisName('x', axisIndex);
    const yAxisName = sliceAxisName('y', axisIndex);
    const xDomain = xDomains[slice.colIndex];
    const yDomain = yDomains[slice.rowIndex];

    layout[xAxisName] = {
      ...axisBase,
      domain: xDomain,
      anchor: yAxisName,
      range: [slice.plane.xBounds.min, slice.plane.xBounds.max],
      title: slice.rowIndex === 2 ? { text: slice.plane.xLabel } : undefined,
    };
    layout[yAxisName] = {
      ...axisBase,
      domain: yDomain,
      anchor: xAxisName,
      range: [slice.plane.yBounds.min, slice.plane.yBounds.max],
      title: slice.colIndex === 0 ? { text: slice.plane.yLabel } : undefined,
      scaleanchor: xAxisName,
      scaleratio: 1,
    };

    layout.annotations = [
      ...((layout.annotations as NonNullable<Layout['annotations']>) ?? []),
      {
        text: formatSliceAnnotation(slice),
        x: (xDomain[0] + xDomain[1]) / 2,
        y: yDomain[1] + 0.018,
        xref: 'paper',
        yref: 'paper',
        xanchor: 'center',
        yanchor: 'bottom',
        showarrow: false,
        align: 'center',
        font: { color: theme.textColor, family: FONT_FAMILY, size: 12 },
      },
    ];
  }

  return layout;
}

export function build2DData(rows: DemoRow[], side: TraceSide): Data[] {
  if (rows.length === 0) {
    return [];
  }

  const x = xAxis(rows);
  const traces: Array<Data & { legend?: LegendId }> = [];

  for (const column of TRACE_EEF_KEYPOINT_COLUMNS) {
    if (!matchesSide(column, side)) {
      continue;
    }

    const values = series(rows, column);
    if (!hasValues(values)) {
      continue;
    }

    traces.push({
      type: 'scatter',
      x,
      y: values,
      mode: 'lines',
      name: build2DLegendLabel(column, side),
      legend: 'legend',
      line: { width: 2.4 },
      yaxis: 'y',
    });
  }

  for (const [distanceColumn, thresholdColumn] of TRACE_TERM_COLUMNS) {
    if (!matchesSide(distanceColumn, side)) {
      continue;
    }

    const values = series(rows, distanceColumn);
    if (!hasValues(values)) {
      continue;
    }

    const traceName = build2DLegendLabel(distanceColumn, side);
    traces.push({
      type: 'scatter',
      x,
      y: values,
      mode: 'lines',
      name: traceName,
      legend: 'legend2',
      line: { width: 2.4 },
      xaxis: 'x2',
      yaxis: 'y2',
    });

    const thresholdValue = rows.find((row) => row[thresholdColumn] != null)?.[thresholdColumn];
    if (typeof thresholdValue === 'number' && x.length > 0) {
      traces.push({
        type: 'scatter',
        x: [x[0], x[x.length - 1]],
        y: [thresholdValue, thresholdValue],
        mode: 'lines',
        name: `${traceName} threshold`,
        legend: 'legend2',
        line: { dash: 'dash', width: 1.6 },
        showlegend: false,
        xaxis: 'x2',
        yaxis: 'y2',
      });
    }
  }

  for (const column of TRACE_Z_COLUMNS) {
    if (!matchesSide(column, side)) {
      continue;
    }

    const values = series(rows, column);
    if (!hasValues(values)) {
      continue;
    }

    traces.push({
      type: 'scatter',
      x,
      y: values,
      mode: 'lines',
      name: build2DLegendLabel(column, side),
      legend: 'legend3',
      line: { width: 2.2 },
      xaxis: 'x3',
      yaxis: 'y3',
    });
  }

  return traces as Data[];
}

export function build2DLayout(_rows: DemoRow[], side: TraceSide): Partial<Layout> {
  const theme = getPlotTheme();
  const axisBase = {
    showgrid: true,
    gridcolor: theme.gridColor,
    zerolinecolor: theme.gridColor,
    color: theme.textColor,
  };
  const sideTitles = SUBPLOT_TITLES.map((title) => `${sideLabel(side)} ${title}`);
  const layout: Partial<Layout> & {
    legend2: Partial<Layout['legend']>;
    legend3: Partial<Layout['legend']>;
  } = {
    template: PLOTLY_WHITE_TEMPLATE,
    height: 980,
    paper_bgcolor: theme.paperBg,
    plot_bgcolor: theme.plotBg,
    font: { color: theme.textColor, family: FONT_FAMILY },
    legend: buildLegendLayout(theme, 0.705),
    legend2: buildLegendLayout(theme, 0.365),
    legend3: buildLegendLayout(theme, -0.06),
    margin: { l: 60, r: 30, t: 60, b: 160 },
    xaxis: { ...axisBase, domain: [0, 1], anchor: 'y', matches: 'x3' },
    yaxis: { ...axisBase, domain: SUBPLOT_DOMAINS[0], title: { text: 'distance [m]' } },
    xaxis2: { ...axisBase, domain: [0, 1], anchor: 'y2', matches: 'x3' },
    yaxis2: { ...axisBase, domain: SUBPLOT_DOMAINS[1], title: { text: 'distance [m]' } },
    xaxis3: { ...axisBase, domain: [0, 1], anchor: 'y3', title: { text: 'episode_step' } },
    yaxis3: { ...axisBase, domain: SUBPLOT_DOMAINS[2], title: { text: 'z [m]' } },
    annotations: sideTitles.map((text, index) => ({
      text,
      x: 0.5,
      y: SUBPLOT_DOMAINS[index][1],
      xref: 'paper',
      yref: 'paper',
      xanchor: 'center',
      yanchor: 'bottom',
      showarrow: false,
      font: { color: theme.textColor, family: FONT_FAMILY, size: 14 },
    })),
  };

  return layout;
}

export function build3DData(rows: DemoRow[]): Data[] {
  if (rows.length === 0) {
    return [];
  }

  return build3DDataForStep(rows, rows.length - 1);
}

export function build3DDataForStep(
  rows: DemoRow[],
  stepIndex: number,
  hiddenLegendGroups?: ReadonlySet<string>,
): Data[] {
  if (rows.length === 0) {
    return [];
  }

  const traces: Data[] = [];
  const clampedStepIndex = Math.max(0, Math.min(stepIndex, rows.length - 1));

  for (const spec of TRACE_3D_SPECS) {
    const points = build3DPoints(rows, spec.prefix);
    if (points.length === 0) {
      continue;
    }

    const legendGroup = spec.prefix;
    const legendVisible = buildTraceVisibility(hiddenLegendGroups, legendGroup, true);
    const groupVisible = buildTraceVisibility(hiddenLegendGroups, legendGroup, false);
    const { past, future, current } = split3DPoints(points, clampedStepIndex);
    const lastPoint = points[points.length - 1];
    const startPoint = points[0];

    if (past.length > 0) {
      const { x, y, z } = coordinates(past);
      traces.push({
        type: 'scatter3d',
        x,
        y,
        z,
        mode: 'lines',
        name: spec.label,
        uid: `${legendGroup}-past`,
        legendgroup: legendGroup,
        visible: legendVisible,
        line: { color: spec.color, dash: spec.dash as Plotly.Dash, width: 6 },
      });
    }

    if (future.length > 1) {
      const { x, y, z } = coordinates(future);
      traces.push({
        type: 'scatter3d',
        x,
        y,
        z,
        mode: 'lines',
        name: `${spec.label} future`,
        uid: `${legendGroup}-future`,
        legendgroup: legendGroup,
        visible: groupVisible,
        line: { color: spec.color, dash: spec.dash as Plotly.Dash, width: 6 },
        opacity: FUTURE_TRACE_OPACITY,
        showlegend: false,
      });
    }

    traces.push({
      type: 'scatter3d',
      x: [startPoint.x],
      y: [startPoint.y],
      z: [startPoint.z],
      mode: 'markers',
      marker: { color: spec.color, size: spec.markerSize, symbol: 'circle' },
      name: `${spec.label} start`,
      uid: `${legendGroup}-start`,
      legendgroup: legendGroup,
      visible: groupVisible,
      showlegend: false,
      opacity: startPoint.rowIndex <= clampedStepIndex ? 1 : FUTURE_MARKER_OPACITY,
    });

    if (current) {
      traces.push({
        type: 'scatter3d',
        x: [current.x],
        y: [current.y],
        z: [current.z],
        mode: 'markers',
        marker: { color: spec.color, size: spec.markerSize + 2, symbol: 'diamond' },
        name: `${spec.label} current`,
        uid: `${legendGroup}-current`,
        legendgroup: legendGroup,
        visible: groupVisible,
        showlegend: false,
      });
    }

    traces.push({
      type: 'scatter3d',
      x: [lastPoint.x],
      y: [lastPoint.y],
      z: [lastPoint.z],
      mode: 'markers',
      marker: { color: spec.color, size: spec.markerSize + 1, symbol: 'x' },
      name: `${spec.label} end`,
      uid: `${legendGroup}-end`,
      legendgroup: legendGroup,
      visible: groupVisible,
      showlegend: false,
      opacity: lastPoint.rowIndex <= clampedStepIndex ? 1 : FUTURE_MARKER_OPACITY,
    });
  }

  return traces;
}

export function build3DLayout(rows: DemoRow[], camera?: PlotSceneCamera | null): Partial<Layout> {
  const theme = getPlotTheme();
  const uirevision = `${rows[0]?.dataset_name ?? 'dataset'}-${rows[0]?.demo_name ?? 'demo'}-3d`;

  return {
    template: PLOTLY_WHITE_TEMPLATE,
    height: 760,
    paper_bgcolor: theme.paperBg,
    plot_bgcolor: theme.plotBg,
    font: { color: theme.textColor, family: FONT_FAMILY },
    uirevision,
    margin: { l: 20, r: 20, t: 30, b: 90 },
    scene: {
      aspectmode: 'data',
      bgcolor: theme.plotBg,
      camera: camera ?? {
        eye: DEFAULT_3D_CAMERA_EYE,
        up: DEFAULT_3D_CAMERA_UP,
      },
      xaxis: {
        title: { text: 'x [m]' },
        backgroundcolor: theme.plotBg,
        gridcolor: theme.gridColor,
        zerolinecolor: theme.gridColor,
        color: theme.textColor,
      },
      yaxis: {
        title: { text: 'y [m]' },
        backgroundcolor: theme.plotBg,
        gridcolor: theme.gridColor,
        zerolinecolor: theme.gridColor,
        color: theme.textColor,
      },
      zaxis: {
        title: { text: 'z [m]' },
        backgroundcolor: theme.plotBg,
        gridcolor: theme.gridColor,
        zerolinecolor: theme.gridColor,
        color: theme.textColor,
      },
    },
    legend: {
      orientation: 'h',
      groupclick: 'togglegroup',
      yanchor: 'top',
      y: -0.12,
      xanchor: 'center',
      x: 0.5,
      bgcolor: theme.legendBg,
      bordercolor: theme.legendBorder,
      borderwidth: 1,
    },
  };
}

export function buildEmptyLayout(
  title: string,
  message: string,
  is3d = false,
): Partial<Layout> {
  const theme = getPlotTheme();
  const base: Partial<Layout> = {
    template: PLOTLY_WHITE_TEMPLATE,
    title: {
      text: title,
      x: 0.02,
      xanchor: 'left',
      font: { color: theme.textColor, family: FONT_FAMILY, size: 20 },
    },
    paper_bgcolor: theme.paperBg,
    plot_bgcolor: theme.plotBg,
    font: { color: theme.textColor, family: FONT_FAMILY },
    annotations: [
      {
        text: message,
        x: 0.5,
        y: 0.5,
        xref: 'paper',
        yref: 'paper',
        showarrow: false,
        font: { color: theme.mutedText, family: FONT_FAMILY, size: 16 },
      },
    ],
  };

  if (is3d) {
    base.scene = {
      xaxis: { visible: false },
      yaxis: { visible: false },
      zaxis: { visible: false },
    };
    base.height = 760;
  } else {
    base.xaxis = { visible: false };
    base.yaxis = { visible: false };
    base.height = 520;
  }

  return base;
}
