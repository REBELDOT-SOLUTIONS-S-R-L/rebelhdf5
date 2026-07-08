import { render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import DatasetProcessingPage from './DatasetProcessingPage';
import { FileService, type H5File, useStore } from './stores';

const mocks = vi.hoisted(() => ({
  pollBackendStatus: vi.fn(),
  scanFiles: vi.fn(),
  runProcess: vi.fn(),
  runLeRobotConvert: vi.fn(),
  getDatasetProcessingInfo: vi.fn(),
  openPoseTraceSource: vi.fn(),
  processDataset: vi.fn(),
}));

vi.mock('./python-backend', () => ({
  pollBackendStatus: mocks.pollBackendStatus,
  scanFiles: mocks.scanFiles,
  runProcess: mocks.runProcess,
  runLeRobotConvert: mocks.runLeRobotConvert,
}));

vi.mock('./pose-trace/hdf5', () => ({
  getDatasetProcessingInfo: mocks.getDatasetProcessingInfo,
  openPoseTraceSource: mocks.openPoseTraceSource,
  processDataset: mocks.processDataset,
}));

interface BackendStatus {
  available: boolean;
  rootDir: string | null;
  version: number | null;
  outputDir?: string;
}

function mockBackend(status: BackendStatus): void {
  mocks.pollBackendStatus.mockImplementation(
    (onStatus: (s: BackendStatus) => void) => {
      onStatus(status);
      return () => {};
    },
  );
}

function remoteFile(name: string): H5File {
  return {
    url: `u-${name}`,
    name,
    service: FileService.Url,
    resolvedUrl: `https://example.com/${name}`,
  };
}

function renderPage(): void {
  render(
    <MemoryRouter initialEntries={['/dataset-processing']}>
      <DatasetProcessingPage />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  useStore.setState({ opened: [] }, false);
  mocks.scanFiles.mockResolvedValue({ files: [], commonKeys: [] });
  mocks.getDatasetProcessingInfo.mockResolvedValue({ keyPaths: [] });
  mocks.openPoseTraceSource.mockResolvedValue({
    sourceId: 's',
    datasetName: 'ds',
    demos: [],
    articulation: null,
    cleanup: () => {},
  });
});

afterEach(() => {
  useStore.setState({ opened: [] }, false);
});

describe('DatasetProcessingPage', () => {
  it('shows the empty state when the backend is unavailable and no file is open', () => {
    mockBackend({ available: false, rootDir: null, version: null });
    renderPage();

    expect(
      screen.getByText(/Open one or more HDF5 files in rebelHDF5/u),
    ).toBeInTheDocument();
    // The backend banner is not shown when the backend is down.
    expect(
      screen.queryByText('Python Processing Server'),
    ).not.toBeInTheDocument();
  });

  it('shows the Python backend banner and a prompt to open files when available', () => {
    mockBackend({
      available: true,
      rootDir: '/data',
      version: 3,
      outputDir: '/out',
    });
    renderPage();

    expect(screen.getByText('Python Processing Server')).toBeInTheDocument();
    expect(
      screen.getByText(
        /Open one or more HDF5 files from the home page to enable processing/u,
      ),
    ).toBeInTheDocument();
  });

  it('lists opened files as source options and the operation choices in backend mode', () => {
    mockBackend({ available: true, rootDir: '/data', version: 3 });
    useStore.setState(
      { opened: [remoteFile('alpha.hdf5'), remoteFile('beta.hdf5')] },
      false,
    );
    renderPage();

    // Operation selector with all four operations.
    const operationSelect = screen.getByLabelText('Operation');
    expect(
      within(operationSelect).getByRole('option', { name: 'Cut' }),
    ).toBeInTheDocument();
    expect(
      within(operationSelect).getByRole('option', { name: 'LeRobot' }),
    ).toBeInTheDocument();

    // Both opened files are offered as sources.
    const sourceSelect = screen.getByLabelText('Source Dataset');
    expect(
      within(sourceSelect).getByRole('option', { name: 'alpha.hdf5' }),
    ).toBeInTheDocument();
    expect(
      within(sourceSelect).getByRole('option', { name: 'beta.hdf5' }),
    ).toBeInTheDocument();
  });
});
