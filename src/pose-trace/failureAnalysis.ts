import type { ObjectDistributionPoint, ObjectDistributionResult } from './types';

const POSITION_BIN_COUNT = 20;
const ROTATION_BIN_COUNT = 20;
const SLICE_BIN_COUNT = 14;
const SLICE_DIVISIONS = 3;
const MAX_RECOMMENDATIONS = 10;
const CI_Z_SCORE = 1.96;
const CONFIDENCE_PRIOR = 6;
const REGION_FAILURE_THRESHOLD = 0.58;
const REGION_CONFIDENCE_THRESHOLD = 0.32;
const REGION_MIN_SCORE = 0.1;

type AxisKey = 'x' | 'y' | 'rotX' | 'rotY';

interface Bounds {
  min: number;
  max: number;
}

interface Bandwidth2D {
  x: number;
  y: number;
}

interface ResetPoint {
  x: number;
  y: number;
  rotX: number;
  rotY: number;
}

interface GeneratedResetPoint extends ResetPoint {
  outcome: 'success' | 'failed';
}

export interface FailureBin {
  xStart: number;
  xEnd: number;
  yStart: number;
  yEnd: number;
  xCenter: number;
  yCenter: number;
  rawSuccessCount: number;
  rawFailedCount: number;
  rawGeneratedCount: number;
  rawTeleopCount: number;
  smoothedSuccessSupport: number;
  smoothedFailedSupport: number;
  smoothedGeneratedSupport: number;
  teleopDensity: number;
  failureRate: number | null;
  confidenceLower: number | null;
  confidenceUpper: number | null;
  confidenceScore: number;
  displayFailureRate: number | null;
  masked: boolean;
}

export interface FailurePlane {
  xLabel: string;
  yLabel: string;
  xBounds: Bounds;
  yBounds: Bounds;
  bandwidth: Bandwidth2D;
  bins: FailureBin[][];
  overlayPoints: Array<{ x: number; y: number }>;
}

export interface FailureSlice {
  rowIndex: number;
  colIndex: number;
  rotXStart: number;
  rotXEnd: number;
  rotYStart: number;
  rotYEnd: number;
  generatedCount: number;
  teleopCount: number;
  plane: FailurePlane;
}

export interface FailureRecommendation {
  score: number;
  rotXStart: number;
  rotXEnd: number;
  rotYStart: number;
  rotYEnd: number;
  xStart: number;
  xEnd: number;
  yStart: number;
  yEnd: number;
  failureRate: number;
  confidenceLower: number;
  confidenceUpper: number;
  confidenceScore: number;
  smoothedGeneratedSupport: number;
  smoothedFailedSupport: number;
  teleopDensity: number;
  cellCount: number;
  sliceRowIndex: number;
  sliceColIndex: number;
}

export interface FailureAnalysisStats {
  successGeneratedCount: number;
  failedGeneratedCount: number;
  analyzedSuccessCount: number;
  analyzedFailedCount: number;
  analyzedGeneratedCount: number;
  skippedGeneratedCount: number;
  teleopCount: number;
  analyzedTeleopCount: number;
  skippedTeleopCount: number;
}

export interface FailureAnalysisResult {
  stats: FailureAnalysisStats;
  positionMap: FailurePlane;
  rotationMap: FailurePlane;
  slices: FailureSlice[];
  recommendations: FailureRecommendation[];
}

export interface BuildFailureAnalysisOptions {
  minGeneratedCount: number;
}

function isFiniteNumber(value: number | null): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function toResetPoint(point: ObjectDistributionPoint): ResetPoint | null {
  if (
    !isFiniteNumber(point.initialX)
    || !isFiniteNumber(point.initialY)
    || !isFiniteNumber(point.initialRx)
    || !isFiniteNumber(point.initialRy)
  ) {
    return null;
  }

  return {
    x: point.initialX,
    y: point.initialY,
    rotX: point.initialRx,
    rotY: point.initialRy,
  };
}

function getAxisValue(point: ResetPoint, axis: AxisKey): number {
  return point[axis];
}

function computeBounds(values: number[]): Bounds {
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (max > min) {
    return { min, max };
  }

  const pad = Math.max(0.01, Math.abs(min) * 0.05, 1);
  return { min: min - pad, max: max + pad };
}

function computeMean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function computeStd(values: number[], mean: number): number {
  if (values.length <= 1) {
    return 0;
  }

  const variance = values.reduce((sum, value) => sum + ((value - mean) ** 2), 0) / (values.length - 1);
  return Math.sqrt(Math.max(variance, 0));
}

function computeBandwidth(values: number[], bounds: Bounds, binCount: number): number {
  const span = Math.max(bounds.max - bounds.min, 1e-6);
  const mean = computeMean(values);
  const std = computeStd(values, mean);
  const scott = std > 0 ? std * (values.length ** (-1 / 6)) : 0;
  const grid = span / Math.max(binCount, 1);
  return Math.max(grid * 1.6, scott, span / 24, 1e-4);
}

function createEdges(bounds: Bounds, binCount: number): number[] {
  const step = (bounds.max - bounds.min) / binCount;
  return Array.from({ length: binCount + 1 }, (_, index) => (
    index === binCount ? bounds.max : bounds.min + step * index
  ));
}

function createCenters(edges: number[]): number[] {
  return Array.from({ length: edges.length - 1 }, (_, index) => (edges[index] + edges[index + 1]) / 2);
}

function findBinIndex(value: number, bounds: Bounds, binCount: number): number | null {
  if (value < bounds.min || value > bounds.max) {
    return null;
  }

  if (value === bounds.max) {
    return binCount - 1;
  }

  const span = bounds.max - bounds.min;
  if (span <= 0) {
    return 0;
  }

  const ratio = (value - bounds.min) / span;
  return Math.max(0, Math.min(binCount - 1, Math.floor(ratio * binCount)));
}

function gaussianWeight(
  x: number,
  y: number,
  centerX: number,
  centerY: number,
  bandwidth: Bandwidth2D,
): number {
  const dx = (x - centerX) / bandwidth.x;
  const dy = (y - centerY) / bandwidth.y;
  return Math.exp(-0.5 * ((dx * dx) + (dy * dy)));
}

function computePosteriorInterval(successSupport: number, failedSupport: number): {
  mean: number;
  lower: number;
  upper: number;
  confidenceScore: number;
} {
  const alpha = failedSupport + 1;
  const beta = successSupport + 1;
  const total = alpha + beta;
  const mean = alpha / total;
  const variance = (alpha * beta) / ((total ** 2) * (total + 1));
  const std = Math.sqrt(Math.max(variance, 0));
  const lower = Math.max(0, mean - (CI_Z_SCORE * std));
  const upper = Math.min(1, mean + (CI_Z_SCORE * std));
  const confidenceScore = (successSupport + failedSupport) / ((successSupport + failedSupport) + CONFIDENCE_PRIOR);
  return { mean, lower, upper, confidenceScore };
}

function buildPlane(
  generatedPoints: GeneratedResetPoint[],
  teleopPoints: ResetPoint[],
  axes: {
    x: AxisKey;
    y: AxisKey;
    xLabel: string;
    yLabel: string;
  },
  binCounts: { x: number; y: number },
  minGeneratedSupport: number,
  bandwidth: Bandwidth2D,
  forcedBounds?: { x: Bounds; y: Bounds },
): FailurePlane {
  const xBounds = forcedBounds?.x ?? computeBounds(generatedPoints.map((point) => getAxisValue(point, axes.x)));
  const yBounds = forcedBounds?.y ?? computeBounds(generatedPoints.map((point) => getAxisValue(point, axes.y)));
  const xEdges = createEdges(xBounds, binCounts.x);
  const yEdges = createEdges(yBounds, binCounts.y);
  const xCenters = createCenters(xEdges);
  const yCenters = createCenters(yEdges);

  const rawCounts = Array.from({ length: binCounts.y }, () => Array.from({ length: binCounts.x }, () => ({
    rawSuccessCount: 0,
    rawFailedCount: 0,
    rawTeleopCount: 0,
  })));

  for (const point of generatedPoints) {
    const xIndex = findBinIndex(getAxisValue(point, axes.x), xBounds, binCounts.x);
    const yIndex = findBinIndex(getAxisValue(point, axes.y), yBounds, binCounts.y);
    if (xIndex == null || yIndex == null) {
      continue;
    }

    if (point.outcome === 'failed') {
      rawCounts[yIndex][xIndex].rawFailedCount += 1;
    } else {
      rawCounts[yIndex][xIndex].rawSuccessCount += 1;
    }
  }

  const overlayPoints = teleopPoints.flatMap((point) => {
    const xValue = getAxisValue(point, axes.x);
    const yValue = getAxisValue(point, axes.y);
    const xIndex = findBinIndex(xValue, xBounds, binCounts.x);
    const yIndex = findBinIndex(yValue, yBounds, binCounts.y);
    if (xIndex == null || yIndex == null) {
      return [];
    }

    rawCounts[yIndex][xIndex].rawTeleopCount += 1;
    return [{ x: xValue, y: yValue }];
  });

  const bins = yCenters.map((centerY, yIndex) => xCenters.map((centerX, xIndex) => {
    let smoothedSuccessSupport = 0;
    let smoothedFailedSupport = 0;
    let teleopDensity = 0;

    for (const point of generatedPoints) {
      const weight = gaussianWeight(
        getAxisValue(point, axes.x),
        getAxisValue(point, axes.y),
        centerX,
        centerY,
        bandwidth,
      );
      if (point.outcome === 'failed') {
        smoothedFailedSupport += weight;
      } else {
        smoothedSuccessSupport += weight;
      }
    }

    for (const point of teleopPoints) {
      teleopDensity += gaussianWeight(
        getAxisValue(point, axes.x),
        getAxisValue(point, axes.y),
        centerX,
        centerY,
        bandwidth,
      );
    }

    const smoothedGeneratedSupport = smoothedSuccessSupport + smoothedFailedSupport;
    const interval = smoothedGeneratedSupport > 0
      ? computePosteriorInterval(smoothedSuccessSupport, smoothedFailedSupport)
      : null;
    const rawSuccessCount = rawCounts[yIndex][xIndex].rawSuccessCount;
    const rawFailedCount = rawCounts[yIndex][xIndex].rawFailedCount;
    const rawGeneratedCount = rawSuccessCount + rawFailedCount;

    return {
      xStart: xEdges[xIndex],
      xEnd: xEdges[xIndex + 1],
      yStart: yEdges[yIndex],
      yEnd: yEdges[yIndex + 1],
      xCenter: centerX,
      yCenter: centerY,
      rawSuccessCount,
      rawFailedCount,
      rawGeneratedCount,
      rawTeleopCount: rawCounts[yIndex][xIndex].rawTeleopCount,
      smoothedSuccessSupport,
      smoothedFailedSupport,
      smoothedGeneratedSupport,
      teleopDensity,
      failureRate: interval?.mean ?? null,
      confidenceLower: interval?.lower ?? null,
      confidenceUpper: interval?.upper ?? null,
      confidenceScore: interval?.confidenceScore ?? 0,
      displayFailureRate: interval ? (0.5 + ((interval.mean - 0.5) * interval.confidenceScore)) : null,
      masked: smoothedGeneratedSupport < minGeneratedSupport,
    } satisfies FailureBin;
  }));

  return {
    xLabel: axes.xLabel,
    yLabel: axes.yLabel,
    xBounds,
    yBounds,
    bandwidth,
    bins,
    overlayPoints,
  };
}

function shouldSeedRegion(bin: FailureBin, minGeneratedSupport: number): boolean {
  if (bin.masked || bin.failureRate == null || bin.confidenceLower == null) {
    return false;
  }

  const regionScore = Math.max(0, bin.confidenceLower - 0.5) * bin.smoothedGeneratedSupport;
  return (
    bin.smoothedGeneratedSupport >= minGeneratedSupport
    && bin.failureRate >= REGION_FAILURE_THRESHOLD
    && bin.confidenceScore >= REGION_CONFIDENCE_THRESHOLD
    && regionScore >= REGION_MIN_SCORE
  );
}

function collectRecommendations(
  slices: FailureSlice[],
  minGeneratedSupport: number,
): FailureRecommendation[] {
  const recommendations: FailureRecommendation[] = [];
  const neighbors = [
    [-1, -1], [-1, 0], [-1, 1],
    [0, -1], [0, 1],
    [1, -1], [1, 0], [1, 1],
  ];

  for (const slice of slices) {
    const height = slice.plane.bins.length;
    const width = slice.plane.bins[0]?.length ?? 0;
    const visited = Array.from({ length: height }, () => Array.from({ length: width }, () => false));

    for (let rowIndex = 0; rowIndex < height; rowIndex += 1) {
      for (let colIndex = 0; colIndex < width; colIndex += 1) {
        const startBin = slice.plane.bins[rowIndex][colIndex];
        if (visited[rowIndex][colIndex] || !shouldSeedRegion(startBin, minGeneratedSupport)) {
          continue;
        }

        const queue: Array<[number, number]> = [[rowIndex, colIndex]];
        const regionBins: FailureBin[] = [];
        visited[rowIndex][colIndex] = true;

        while (queue.length > 0) {
          const [currentRow, currentCol] = queue.shift()!;
          const currentBin = slice.plane.bins[currentRow][currentCol];
          regionBins.push(currentBin);

          for (const [rowDelta, colDelta] of neighbors) {
            const nextRow = currentRow + rowDelta;
            const nextCol = currentCol + colDelta;
            if (
              nextRow < 0
              || nextCol < 0
              || nextRow >= height
              || nextCol >= width
              || visited[nextRow][nextCol]
            ) {
              continue;
            }

            const nextBin = slice.plane.bins[nextRow][nextCol];
            if (!shouldSeedRegion(nextBin, minGeneratedSupport)) {
              continue;
            }

            visited[nextRow][nextCol] = true;
            queue.push([nextRow, nextCol]);
          }
        }

        const smoothedGeneratedSupport = regionBins.reduce((sum, bin) => sum + bin.smoothedGeneratedSupport, 0);
        const smoothedFailedSupport = regionBins.reduce((sum, bin) => sum + bin.smoothedFailedSupport, 0);
        const teleopDensity = regionBins.reduce((sum, bin) => sum + bin.teleopDensity, 0) / regionBins.length;
        const interval = computePosteriorInterval(
          smoothedGeneratedSupport - smoothedFailedSupport,
          smoothedFailedSupport,
        );
        const score = Math.max(0, interval.lower - 0.5) * smoothedGeneratedSupport / (1 + teleopDensity);

        if (regionBins.length < 2 && smoothedGeneratedSupport < (minGeneratedSupport * 1.8)) {
          continue;
        }

        recommendations.push({
          score,
          rotXStart: slice.rotXStart,
          rotXEnd: slice.rotXEnd,
          rotYStart: slice.rotYStart,
          rotYEnd: slice.rotYEnd,
          xStart: Math.min(...regionBins.map((bin) => bin.xStart)),
          xEnd: Math.max(...regionBins.map((bin) => bin.xEnd)),
          yStart: Math.min(...regionBins.map((bin) => bin.yStart)),
          yEnd: Math.max(...regionBins.map((bin) => bin.yEnd)),
          failureRate: interval.mean,
          confidenceLower: interval.lower,
          confidenceUpper: interval.upper,
          confidenceScore: interval.confidenceScore,
          smoothedGeneratedSupport,
          smoothedFailedSupport,
          teleopDensity,
          cellCount: regionBins.length,
          sliceRowIndex: slice.rowIndex,
          sliceColIndex: slice.colIndex,
        });
      }
    }
  }

  return recommendations
    .sort((left, right) => (
      right.score - left.score
      || right.confidenceLower - left.confidenceLower
      || right.smoothedFailedSupport - left.smoothedFailedSupport
      || left.teleopDensity - right.teleopDensity
    ))
    .slice(0, MAX_RECOMMENDATIONS);
}

export function buildFailureAnalysis(
  result: ObjectDistributionResult,
  options: BuildFailureAnalysisOptions,
): FailureAnalysisResult | null {
  const minGeneratedSupport = Math.max(1, Math.trunc(options.minGeneratedCount) || 1);
  const successGeneratedPoints: GeneratedResetPoint[] = [];
  const failedGeneratedPoints: GeneratedResetPoint[] = [];
  const teleopPoints: ResetPoint[] = [];

  for (const point of result.successPoints) {
    const resetPoint = toResetPoint(point);
    if (!resetPoint) {
      continue;
    }

    successGeneratedPoints.push({ ...resetPoint, outcome: 'success' });
  }

  for (const point of result.failedPoints) {
    const resetPoint = toResetPoint(point);
    if (!resetPoint) {
      continue;
    }

    failedGeneratedPoints.push({ ...resetPoint, outcome: 'failed' });
  }

  for (const point of result.teleopPoints) {
    const resetPoint = toResetPoint(point);
    if (!resetPoint) {
      continue;
    }

    teleopPoints.push(resetPoint);
  }

  const generatedPoints = [...successGeneratedPoints, ...failedGeneratedPoints];
  if (generatedPoints.length === 0) {
    return null;
  }

  const positionBounds = {
    x: computeBounds(generatedPoints.map((point) => point.x)),
    y: computeBounds(generatedPoints.map((point) => point.y)),
  };
  const rotationBounds = {
    x: computeBounds(generatedPoints.map((point) => point.rotX)),
    y: computeBounds(generatedPoints.map((point) => point.rotY)),
  };
  const positionBandwidth = {
    x: computeBandwidth(generatedPoints.map((point) => point.x), positionBounds.x, POSITION_BIN_COUNT),
    y: computeBandwidth(generatedPoints.map((point) => point.y), positionBounds.y, POSITION_BIN_COUNT),
  };
  const rotationBandwidth = {
    x: computeBandwidth(generatedPoints.map((point) => point.rotX), rotationBounds.x, ROTATION_BIN_COUNT),
    y: computeBandwidth(generatedPoints.map((point) => point.rotY), rotationBounds.y, ROTATION_BIN_COUNT),
  };

  const positionMap = buildPlane(
    generatedPoints,
    teleopPoints,
    {
      x: 'x',
      y: 'y',
      xLabel: 'Initial x [m]',
      yLabel: 'Initial y [m]',
    },
    { x: POSITION_BIN_COUNT, y: POSITION_BIN_COUNT },
    minGeneratedSupport,
    positionBandwidth,
    positionBounds,
  );

  const rotationMap = buildPlane(
    generatedPoints,
    teleopPoints,
    {
      x: 'rotX',
      y: 'rotY',
      xLabel: 'Initial rot x [deg]',
      yLabel: 'Initial rot y [deg]',
    },
    { x: ROTATION_BIN_COUNT, y: ROTATION_BIN_COUNT },
    minGeneratedSupport,
    rotationBandwidth,
    rotationBounds,
  );

  const sliceXEdges = createEdges(rotationBounds.x, SLICE_DIVISIONS);
  const sliceYEdges = createEdges(rotationBounds.y, SLICE_DIVISIONS);
  const generatedSlices = Array.from(
    { length: SLICE_DIVISIONS * SLICE_DIVISIONS },
    () => [] as GeneratedResetPoint[],
  );
  const teleopSlices = Array.from(
    { length: SLICE_DIVISIONS * SLICE_DIVISIONS },
    () => [] as ResetPoint[],
  );

  for (const point of generatedPoints) {
    const xIndex = findBinIndex(point.rotX, rotationBounds.x, SLICE_DIVISIONS);
    const yIndex = findBinIndex(point.rotY, rotationBounds.y, SLICE_DIVISIONS);
    if (xIndex == null || yIndex == null) {
      continue;
    }

    generatedSlices[(yIndex * SLICE_DIVISIONS) + xIndex].push(point);
  }

  for (const point of teleopPoints) {
    const xIndex = findBinIndex(point.rotX, rotationBounds.x, SLICE_DIVISIONS);
    const yIndex = findBinIndex(point.rotY, rotationBounds.y, SLICE_DIVISIONS);
    if (xIndex == null || yIndex == null) {
      continue;
    }

    teleopSlices[(yIndex * SLICE_DIVISIONS) + xIndex].push(point);
  }

  const slices: FailureSlice[] = [];
  for (let rowIndex = 0; rowIndex < SLICE_DIVISIONS; rowIndex += 1) {
    for (let colIndex = 0; colIndex < SLICE_DIVISIONS; colIndex += 1) {
      const sliceIndex = (rowIndex * SLICE_DIVISIONS) + colIndex;
      const sliceGeneratedPoints = generatedSlices[sliceIndex];
      const sliceTeleopPoints = teleopSlices[sliceIndex];
      const plane = buildPlane(
        sliceGeneratedPoints,
        sliceTeleopPoints,
        {
          x: 'x',
          y: 'y',
          xLabel: 'Initial x [m]',
          yLabel: 'Initial y [m]',
        },
        { x: SLICE_BIN_COUNT, y: SLICE_BIN_COUNT },
        minGeneratedSupport,
        positionBandwidth,
        positionBounds,
      );

      slices.push({
        rowIndex,
        colIndex,
        rotXStart: sliceXEdges[colIndex],
        rotXEnd: sliceXEdges[colIndex + 1],
        rotYStart: sliceYEdges[rowIndex],
        rotYEnd: sliceYEdges[rowIndex + 1],
        generatedCount: sliceGeneratedPoints.length,
        teleopCount: sliceTeleopPoints.length,
        plane,
      });
    }
  }

  return {
    stats: {
      successGeneratedCount: result.successPoints.length,
      failedGeneratedCount: result.failedPoints.length,
      analyzedSuccessCount: successGeneratedPoints.length,
      analyzedFailedCount: failedGeneratedPoints.length,
      analyzedGeneratedCount: generatedPoints.length,
      skippedGeneratedCount: (result.successPoints.length + result.failedPoints.length) - generatedPoints.length,
      teleopCount: result.teleopPoints.length,
      analyzedTeleopCount: teleopPoints.length,
      skippedTeleopCount: result.teleopPoints.length - teleopPoints.length,
    },
    positionMap,
    rotationMap,
    slices,
    recommendations: collectRecommendations(slices, minGeneratedSupport),
  };
}
