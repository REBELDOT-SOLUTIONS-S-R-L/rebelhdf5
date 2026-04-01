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

const PAPER_BG = '#ffffff';
const PLOT_BG = '#f7f7fa';
const GRID_COLOR = 'rgba(90, 90, 102, 0.14)';
const TEXT_COLOR = '#5a5a66';
const MUTED_TEXT = '#7b7b87';
const FONT_FAMILY = 'Roboto, sans-serif';
const PLOTLY_WHITE_TEMPLATE = 'plotly_white' as unknown as Layout['template'];

const SUBPLOT_DOMAINS: [number, number][] = [
  [0.72, 1.0],
  [0.38, 0.64],
  [0.0, 0.28],
];

const SUBPLOT_TITLES = [
  'EEF To Keypoint Distances',
  'Garment Fold-Term Distances',
  'EEF And Garment Keypoint Heights',
];

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

function buildTitle(rows: DemoRow[]): string {
  if (rows.length === 0) {
    return 'Pose Trace';
  }

  const first = rows[0];
  return `${first.dataset_name} | ${first.demo_name} | steps=${rows.length}`;
}

function matchesSide(column: string, side: TraceSide): boolean {
  return column.includes(`_${side}_`);
}

function sideLabel(side: TraceSide): string {
  return side === 'left' ? 'Left EEF' : 'Right EEF';
}

function build2DPanelTitle(rows: DemoRow[], side: TraceSide): string {
  if (rows.length === 0) {
    return `${sideLabel(side)} 2D traces`;
  }

  return `${rows[0].demo_name} | ${sideLabel(side)}`;
}

export function build2DData(rows: DemoRow[], side: TraceSide): Data[] {
  if (rows.length === 0) {
    return [];
  }

  const x = xAxis(rows);
  const traces: Data[] = [];

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
      name: humanizeColumnName(column),
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

    const traceName = humanizeColumnName(distanceColumn);
    traces.push({
      type: 'scatter',
      x,
      y: values,
      mode: 'lines',
      name: traceName,
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
      name: humanizeColumnName(column),
      line: { width: 2.2 },
      xaxis: 'x3',
      yaxis: 'y3',
    });
  }

  return traces;
}

export function build2DLayout(rows: DemoRow[], side: TraceSide): Partial<Layout> {
  const axisBase = {
    showgrid: true,
    gridcolor: GRID_COLOR,
    zerolinecolor: GRID_COLOR,
    color: TEXT_COLOR,
  };
  const sideTitles = SUBPLOT_TITLES.map((title) => `${sideLabel(side)} ${title}`);
  const legend = {
    title: { text: sideLabel(side), font: { color: TEXT_COLOR } },
    orientation: 'h',
    entrywidthmode: 'fraction' as const,
    entrywidth: 0.32,
    yanchor: 'top' as const,
    y: -0.16,
    xanchor: 'left' as const,
    x: 0.0,
    bgcolor: 'rgba(255, 255, 255, 0.92)',
    bordercolor: 'rgba(90, 90, 102, 0.14)',
    borderwidth: 1,
  } as Partial<Layout['legend']> & {
    entrywidthmode: 'fraction';
    entrywidth: number;
  };

  return {
    template: PLOTLY_WHITE_TEMPLATE,
    title: {
      text: build2DPanelTitle(rows, side),
      x: 0.02,
      xanchor: 'left',
      font: { color: TEXT_COLOR, family: FONT_FAMILY, size: 20 },
    },
    height: 940,
    paper_bgcolor: PAPER_BG,
    plot_bgcolor: PLOT_BG,
    font: { color: TEXT_COLOR, family: FONT_FAMILY },
    legend,
    margin: { l: 60, r: 30, t: 90, b: 130 },
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
      font: { color: TEXT_COLOR, family: FONT_FAMILY, size: 15 },
    })),
  };
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

export function build3DLayout(rows: DemoRow[]): Partial<Layout> {
  return {
    template: PLOTLY_WHITE_TEMPLATE,
    title: {
      text: rows.length > 0 ? `${buildTitle(rows)} | 3D trajectories` : '3D Pose Trace',
      x: 0.02,
      xanchor: 'left',
      font: { color: TEXT_COLOR, family: FONT_FAMILY, size: 20 },
    },
    height: 760,
    paper_bgcolor: PAPER_BG,
    plot_bgcolor: PLOT_BG,
    font: { color: TEXT_COLOR, family: FONT_FAMILY },
    margin: { l: 20, r: 20, t: 80, b: 20 },
    scene: {
      aspectmode: 'data',
      bgcolor: PLOT_BG,
      xaxis: {
        title: { text: 'x [m]' },
        backgroundcolor: PLOT_BG,
        gridcolor: GRID_COLOR,
        zerolinecolor: GRID_COLOR,
        color: TEXT_COLOR,
      },
      yaxis: {
        title: { text: 'y [m]' },
        backgroundcolor: PLOT_BG,
        gridcolor: GRID_COLOR,
        zerolinecolor: GRID_COLOR,
        color: TEXT_COLOR,
      },
      zaxis: {
        title: { text: 'z [m]' },
        backgroundcolor: PLOT_BG,
        gridcolor: GRID_COLOR,
        zerolinecolor: GRID_COLOR,
        color: TEXT_COLOR,
      },
    },
    legend: {
      orientation: 'h',
      yanchor: 'bottom',
      y: 1.02,
      xanchor: 'left',
      x: 0.0,
      bgcolor: 'rgba(255, 255, 255, 0.92)',
      bordercolor: 'rgba(90, 90, 102, 0.14)',
      borderwidth: 1,
    },
  };
}

export function buildEmptyLayout(
  title: string,
  message: string,
  is3d = false,
): Partial<Layout> {
  const base: Partial<Layout> = {
    template: PLOTLY_WHITE_TEMPLATE,
    title: {
      text: title,
      x: 0.02,
      xanchor: 'left',
      font: { color: TEXT_COLOR, family: FONT_FAMILY, size: 20 },
    },
    paper_bgcolor: PAPER_BG,
    plot_bgcolor: PLOT_BG,
    font: { color: TEXT_COLOR, family: FONT_FAMILY },
    annotations: [
      {
        text: message,
        x: 0.5,
        y: 0.5,
        xref: 'paper',
        yref: 'paper',
        showarrow: false,
        font: { color: MUTED_TEXT, family: FONT_FAMILY, size: 16 },
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
