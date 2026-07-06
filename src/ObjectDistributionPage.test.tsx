import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import ObjectDistributionPage from './ObjectDistributionPage';
import { FileService, type H5File, useStore } from './stores';

const mocks = vi.hoisted(() => ({
  loadObjectDistribution: vi.fn(),
  openPoseTraceSource: vi.fn(),
}));

vi.mock('./pose-trace/hdf5', () => ({
  loadObjectDistribution: mocks.loadObjectDistribution,
  openPoseTraceSource: mocks.openPoseTraceSource,
}));

vi.mock('./pose-trace/PlotlyChart', () => ({ default: () => null }));

vi.mock('./pose-trace/plotConfig', () => ({
  buildObjectDistributionData: () => [],
  buildObjectDistributionLayout: () => ({}),
  buildEmptyLayout: () => ({}),
  buildFailureMapData: () => [],
  buildFailureMapLayout: () => ({}),
  buildFailureSliceData: () => [],
  buildFailureSliceLayout: () => ({}),
}));

vi.mock('./pose-trace/failureAnalysis', () => ({
  buildFailureAnalysis: () => null,
}));

function remoteFile(name: string, url = `u-${name}`): H5File {
  return {
    url,
    name,
    service: FileService.Url,
    resolvedUrl: `https://example.com/${name}`,
  };
}

const successF = remoteFile('cube_generated_a.hdf5', 'u-success');
const failedF = remoteFile('cube_generated_a_failed.hdf5', 'u-failed');
const teleopF = remoteFile('cube_annotated_a.hdf5', 'u-teleop');

function renderPage(): void {
  render(
    <MemoryRouter initialEntries={['/object-distribution']}>
      <ObjectDistributionPage />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  useStore.setState({ opened: [] }, false);
  mocks.openPoseTraceSource.mockResolvedValue({
    sourceId: 's',
    datasetName: 'ds',
    demos: [],
    articulation: null,
    cleanup: vi.fn(),
  });
  mocks.loadObjectDistribution.mockResolvedValue({
    anchor: 'initial',
    successPoints: [],
    failedPoints: [],
    teleopPoints: [],
    teleopDiagnostics: null,
    objectNames: [],
  });
});

afterEach(() => {
  useStore.setState({ opened: [] }, false);
});

describe('ObjectDistributionPage', () => {
  it('shows the empty state when no files are open', () => {
    renderPage();
    expect(
      screen.getByText(/Open HDF5 files in rebelHDF5/u),
    ).toBeInTheDocument();
  });

  it('auto-selects success/failed/teleop datasets from a coherent pack', async () => {
    useStore.setState({ opened: [successF, failedF, teleopF] }, false);
    renderPage();

    await waitFor(() => {
      expect(screen.getByLabelText('Successful Generated Dataset')).toHaveValue(
        'u-success',
      );
    });
    expect(screen.getByLabelText('Failed Generated Dataset')).toHaveValue(
      'u-failed',
    );
    expect(screen.getByLabelText('Teleop Dataset')).toHaveValue('u-teleop');
  });

  it('switches the active analysis tab on click', async () => {
    useStore.setState({ opened: [successF, failedF, teleopF] }, false);
    const user = userEvent.setup();
    renderPage();

    const scatterTab = await screen.findByRole('tab', { name: 'Scatter' });
    const positionTab = screen.getByRole('tab', { name: 'Position Map' });
    expect(scatterTab).toHaveAttribute('aria-selected', 'true');

    await user.click(positionTab);
    expect(positionTab).toHaveAttribute('aria-selected', 'true');
    expect(scatterTab).toHaveAttribute('aria-selected', 'false');
  });
});
