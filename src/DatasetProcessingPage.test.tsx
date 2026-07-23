import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
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
  Reflect.deleteProperty(globalThis, 'rebelHdf5Desktop');
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

  it('defaults LeRobot conversion to v3 and H.264 and updates the output name', async () => {
    const user = userEvent.setup();
    mockBackend({ available: true, rootDir: '/data', version: 10 });
    useStore.setState({ opened: [remoteFile('run.hdf5')] }, false);
    renderPage();

    await user.selectOptions(screen.getByLabelText('Operation'), 'lerobot');

    expect(screen.getByLabelText('Output Format')).toHaveValue('v3.0');
    expect(screen.getByLabelText('Video Codec')).toHaveValue('h264');
    expect(screen.getByText('run-lerobot-v3')).toBeInTheDocument();
    expect(screen.getByLabelText('Modality JSON')).toHaveValue('');

    await user.selectOptions(screen.getByLabelText('Output Format'), 'v2.1');
    expect(screen.getByText('run-lerobot-v21')).toBeInTheDocument();
    expect(screen.getByText(/V2\.1 keeps one Parquet/u)).toBeInTheDocument();
  });

  it('sends the selected format and displays conversion warnings', async () => {
    const user = userEvent.setup();
    mockBackend({ available: true, rootDir: '/data', version: 10 });
    useStore.setState({ opened: [remoteFile('run.hdf5')] }, false);
    mocks.scanFiles.mockResolvedValue({
      files: [
        {
          name: 'run.hdf5',
          path: 'run.hdf5',
          demoCount: 1,
          demoNames: ['demo_0'],
          keys: [],
        },
      ],
      commonKeys: [],
    });
    mocks.runLeRobotConvert.mockImplementation(async (_request, callbacks) => {
      callbacks.onWarning?.('NVENC failed; using CPU H.264.');
      return {
        fileName: 'run-lerobot-v3',
        demoCount: 1,
        selectedKeyCount: 0,
        fileSize: 1,
        outputType: 'directory',
      };
    });
    renderPage();

    await user.selectOptions(screen.getByLabelText('Operation'), 'lerobot');
    await user.type(
      screen.getByLabelText('Modality JSON'),
      '/tmp/modality.json',
    );
    await user.type(
      screen.getByLabelText('Output Parent Folder'),
      '/chosen-output',
    );
    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Convert' })).toBeEnabled();
    });
    await user.click(screen.getByRole('button', { name: 'Convert' }));

    expect(
      await screen.findByText('NVENC failed; using CPU H.264.'),
    ).toBeInTheDocument();
    expect(mocks.runLeRobotConvert).toHaveBeenCalledWith(
      expect.objectContaining({
        outputVersion: 'v3.0',
        videoCodec: 'h264',
        outputDirectory: '/chosen-output',
      }),
      expect.objectContaining({
        onProgress: expect.any(Function),
        onWarning: expect.any(Function),
      }),
    );
  });

  it('chooses a LeRobot output folder with the desktop picker', async () => {
    const user = userEvent.setup();
    const chooseDirectory = vi.fn().mockResolvedValue('/picked-output');
    globalThis.rebelHdf5Desktop = { chooseDirectory };
    mockBackend({
      available: true,
      rootDir: '/data',
      version: 10,
      outputDir: '/default-output',
    });
    useStore.setState({ opened: [remoteFile('run.hdf5')] }, false);
    renderPage();

    await user.selectOptions(screen.getByLabelText('Operation'), 'lerobot');
    await user.click(screen.getByRole('button', { name: 'Browse' }));

    expect(chooseDirectory).toHaveBeenCalledWith('/default-output');
    expect(screen.getByLabelText('Output Parent Folder')).toHaveValue(
      '/picked-output',
    );
  });
});
