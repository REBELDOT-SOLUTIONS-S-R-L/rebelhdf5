import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import PoseTracePage from './PoseTracePage';
import { FileService, type H5File, useStore } from './stores';

const mocks = vi.hoisted(() => ({
  openPoseTraceSource: vi.fn(),
  loadDemoRows: vi.fn(),
}));

vi.mock('./pose-trace/hdf5', () => ({
  openPoseTraceSource: mocks.openPoseTraceSource,
  loadDemoRows: mocks.loadDemoRows,
}));

vi.mock('./pose-trace/PlotlyChart', () => ({ default: () => null }));

vi.mock('./pose-trace/plotConfig', () => ({
  build3DDataForStep: () => [],
  build3DLayout: () => ({}),
  buildCombinedJointChartData: () => [],
  buildEmptyLayout: () => ({}),
  buildJointChartData: () => [],
  buildJointChartLayout: () => ({}),
  getDefaultHidden3DTraceGroups: () => [],
  getJointChartSpecs: () => [],
}));

function localFile(): H5File {
  return {
    url: 'u-trace.hdf5',
    name: 'trace.hdf5',
    service: FileService.Local,
    resolvedUrl: 'blob:trace.hdf5',
    serverPath: '/data/trace.hdf5',
    file: new File([new Uint8Array([1])], 'trace.hdf5'),
  };
}

function renderPage(url = '/pose-trace'): void {
  render(
    <MemoryRouter initialEntries={[url]}>
      <PoseTracePage />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  useStore.setState({ opened: [] }, false);
  mocks.openPoseTraceSource.mockResolvedValue({
    sourceId: 's1',
    datasetName: 'trace',
    demos: [
      {
        name: 'demo_0',
        episode_index: 0,
        num_samples: 3,
        success: true,
        source_episode_index: 0,
      },
    ],
    articulation: null,
    cleanup: vi.fn(),
  });
  mocks.loadDemoRows.mockResolvedValue([{}, {}, {}]);
});

afterEach(() => {
  useStore.setState({ opened: [] }, false);
});

describe('PoseTracePage', () => {
  it('shows the empty state when no file is open', () => {
    renderPage();
    expect(
      screen.getByText(/Open an HDF5 file in myHDF5/u),
    ).toBeInTheDocument();
  });

  it('opens the source, loads demo rows and renders the step slider', async () => {
    const file = localFile();
    useStore.setState({ opened: [file] }, false);
    renderPage(`/pose-trace?url=${encodeURIComponent(file.url)}`);

    // Demo dropdown appears after the (mocked) source opens.
    const demoSelect = await screen.findByLabelText('Demo');
    expect(demoSelect).toBeInTheDocument();
    expect(mocks.openPoseTraceSource).toHaveBeenCalledWith(file);

    // Rows load for the auto-selected demo, enabling the 3D step slider.
    const slider = await screen.findByRole('slider');
    expect(mocks.loadDemoRows).toHaveBeenCalledWith(
      expect.objectContaining({ sourceId: 's1' }),
      'demo_0',
    );
    // Three rows → slider spans indices 0..2.
    expect(slider).toHaveAttribute('max', '2');
  });
});
