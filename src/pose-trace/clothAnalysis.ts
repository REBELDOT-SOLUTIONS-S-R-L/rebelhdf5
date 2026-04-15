import type { ClothDistributionPoint, ClothDistributionResult } from './types';

const POSITION_BIN_COUNT = 24;
const ROTATION_BIN_COUNT = 24;
const SLICE_BIN_COUNT = 18;
const SLICE_DIVISIONS = 3;
const MAX_RECOMMENDATIONS = 10;

type AxisKey = 'x' | 'y' | 'rotX' | 'rotY';

interface Bounds {
  min: number;
  max: number;
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
  successCount: number;
  failedCount: number;
  totalGeneratedCount: number;
  teleopCount: number;
  failureRate: number | null;
  masked: boolean;
}

export interface FailurePlane {
  xLabel: string;
  yLabel: string;
  xBounds: Bounds;
  yBounds: Bounds;
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
  successCount: number;
  failedCount: number;
  totalGeneratedCount: number;
  teleopCount: number;
  failureRate: number;
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

function toResetPoint(point: ClothDistributionPoint): ResetPoint | null {
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

function createEdges(bounds: Bounds, binCount: number): number[] {
  const step = (bounds.max - bounds.min) / binCount;
  return Array.from({ length: binCount + 1 }, (_, index) => (
    index === binCount ? bounds.max : bounds.min + step * index
  ));
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
  minGeneratedCount: number,
  forcedBounds?: { x: Bounds; y: Bounds },
): FailurePlane {
  const xBounds = forcedBounds?.x ?? computeBounds(generatedPoints.map((point) => getAxisValue(point, axes.x)));
  const yBounds = forcedBounds?.y ?? computeBounds(generatedPoints.map((point) => getAxisValue(point, axes.y)));
  const xEdges = createEdges(xBounds, binCounts.x);
  const yEdges = createEdges(yBounds, binCounts.y);

  const counts = Array.from({ length: binCounts.y }, () => Array.from({ length: binCounts.x }, () => ({
    successCount: 0,
    failedCount: 0,
    teleopCount: 0,
  })));

  for (const point of generatedPoints) {
    const xIndex = findBinIndex(getAxisValue(point, axes.x), xBounds, binCounts.x);
    const yIndex = findBinIndex(getAxisValue(point, axes.y), yBounds, binCounts.y);
    if (xIndex == null || yIndex == null) {
      continue;
    }

    if (point.outcome === 'failed') {
      counts[yIndex][xIndex].failedCount += 1;
    } else {
      counts[yIndex][xIndex].successCount += 1;
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

    counts[yIndex][xIndex].teleopCount += 1;
    return [{ x: xValue, y: yValue }];
  });

  const bins = counts.map((row, yIndex) => row.map((count, xIndex) => {
    const totalGeneratedCount = count.successCount + count.failedCount;
    return {
      xStart: xEdges[xIndex],
      xEnd: xEdges[xIndex + 1],
      yStart: yEdges[yIndex],
      yEnd: yEdges[yIndex + 1],
      xCenter: (xEdges[xIndex] + xEdges[xIndex + 1]) / 2,
      yCenter: (yEdges[yIndex] + yEdges[yIndex + 1]) / 2,
      successCount: count.successCount,
      failedCount: count.failedCount,
      totalGeneratedCount,
      teleopCount: count.teleopCount,
      failureRate: totalGeneratedCount > 0 ? count.failedCount / totalGeneratedCount : null,
      masked: totalGeneratedCount < minGeneratedCount,
    } satisfies FailureBin;
  }));

  return {
    xLabel: axes.xLabel,
    yLabel: axes.yLabel,
    xBounds,
    yBounds,
    bins,
    overlayPoints,
  };
}

function collectRecommendations(
  slices: FailureSlice[],
  minGeneratedCount: number,
): FailureRecommendation[] {
  const recommendations: FailureRecommendation[] = [];

  for (const slice of slices) {
    for (const row of slice.plane.bins) {
      for (const bin of row) {
        if (bin.totalGeneratedCount < minGeneratedCount || bin.failedCount === 0 || bin.failureRate == null) {
          continue;
        }

        recommendations.push({
          score: (bin.failureRate * bin.failedCount) / (1 + bin.teleopCount),
          rotXStart: slice.rotXStart,
          rotXEnd: slice.rotXEnd,
          rotYStart: slice.rotYStart,
          rotYEnd: slice.rotYEnd,
          xStart: bin.xStart,
          xEnd: bin.xEnd,
          yStart: bin.yStart,
          yEnd: bin.yEnd,
          successCount: bin.successCount,
          failedCount: bin.failedCount,
          totalGeneratedCount: bin.totalGeneratedCount,
          teleopCount: bin.teleopCount,
          failureRate: bin.failureRate,
          sliceRowIndex: slice.rowIndex,
          sliceColIndex: slice.colIndex,
        });
      }
    }
  }

  return recommendations
    .sort((left, right) => (
      right.score - left.score
      || right.failedCount - left.failedCount
      || right.failureRate - left.failureRate
      || left.teleopCount - right.teleopCount
    ))
    .slice(0, MAX_RECOMMENDATIONS);
}

export function buildFailureAnalysis(
  result: ClothDistributionResult,
  options: BuildFailureAnalysisOptions,
): FailureAnalysisResult | null {
  const minGeneratedCount = Math.max(1, Math.trunc(options.minGeneratedCount) || 1);
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
    minGeneratedCount,
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
    minGeneratedCount,
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
        minGeneratedCount,
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
    recommendations: collectRecommendations(slices, minGeneratedCount),
  };
}
