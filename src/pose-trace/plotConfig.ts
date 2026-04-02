import type Plotly from 'plotly.js';
import type { Data, Layout } from 'plotly.js';

import {
  TRACE_3D_SPECS,
  TRACE_EEF_KEYPOINT_COLUMNS,
  TRACE_TERM_COLUMNS,
  TRACE_Z_COLUMNS,
  humanizeColumnName,
} from './schema';
import type { DemoRow } from './types';

export type TraceSide = 'left' | 'right';
type LegendId = 'legend' | 'legend2' | 'legend3';

const FONT_FAMILY = 'Roboto, sans-serif';
const PLOTLY_WHITE_TEMPLATE = 'plotly_white' as unknown as Layout['template'];

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

  const traces: Data[] = [];

  for (const spec of TRACE_3D_SPECS) {
    const xs: number[] = [];
    const ys: number[] = [];
    const zs: number[] = [];

    for (const row of rows) {
      const x = row[`${spec.prefix}_x`];
      const y = row[`${spec.prefix}_y`];
      const z = row[`${spec.prefix}_z`];

      if (
        typeof x !== 'number' ||
        typeof y !== 'number' ||
        typeof z !== 'number'
      ) {
        continue;
      }

      xs.push(x);
      ys.push(y);
      zs.push(z);
    }

    if (xs.length === 0) {
      continue;
    }

    traces.push({
      type: 'scatter3d',
      x: xs,
      y: ys,
      z: zs,
      mode: 'lines',
      name: spec.label,
      line: { color: spec.color, dash: spec.dash as Plotly.Dash, width: 6 },
    });
    traces.push({
      type: 'scatter3d',
      x: [xs[0]],
      y: [ys[0]],
      z: [zs[0]],
      mode: 'markers',
      marker: { color: spec.color, size: spec.markerSize, symbol: 'circle' },
      name: `${spec.label} start`,
      showlegend: false,
    });
    traces.push({
      type: 'scatter3d',
      x: [xs[xs.length - 1]],
      y: [ys[ys.length - 1]],
      z: [zs[zs.length - 1]],
      mode: 'markers',
      marker: { color: spec.color, size: spec.markerSize + 1, symbol: 'x' },
      name: `${spec.label} end`,
      showlegend: false,
    });
  }

  return traces;
}

export function build3DLayout(_rows: DemoRow[]): Partial<Layout> {
  const theme = getPlotTheme();
  return {
    template: PLOTLY_WHITE_TEMPLATE,
    height: 760,
    paper_bgcolor: theme.paperBg,
    plot_bgcolor: theme.plotBg,
    font: { color: theme.textColor, family: FONT_FAMILY },
    margin: { l: 20, r: 20, t: 30, b: 90 },
    scene: {
      aspectmode: 'data',
      bgcolor: theme.plotBg,
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
