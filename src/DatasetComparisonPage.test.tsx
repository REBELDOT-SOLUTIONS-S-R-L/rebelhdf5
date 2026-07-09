import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import DatasetComparisonPage from './DatasetComparisonPage';
import {
  getDatasetProcessingInfo,
  loadDatasetComparisonValues,
  openPoseTraceSource,
} from './pose-trace/hdf5';
import type {
  DatasetComparisonValuesResult,
  DatasetProcessingSourceInfo,
  DemoInfo,
  PoseTraceSource,
} from './pose-trace/types';
import { FileService, type H5File, useStore } from './stores';

vi.mock('./pose-trace/hdf5', () => ({
  getDatasetProcessingInfo: vi.fn(),
  loadDatasetComparisonValues: vi.fn(),
  openPoseTraceSource: vi.fn(),
}));

const initialState = useStore.getState();

interface MockDatasetSource {
  demos: string[];
  keys: string[];
  values: Record<string, unknown>;
}

const sourceData = new Map<string, MockDatasetSource>();

function makeRemote(url: string, name = url): H5File {
  return {
    url,
    name,
    service: FileService.Url,
    resolvedUrl: url,
  };
}

function demo(name: string): DemoInfo {
  return {
    name,
    episode_index: null,
    num_samples: null,
    success: null,
    source_episode_index: null,
  };
}

function renderPage(
  initialEntry = '/dataset-comparison?url=https%3A%2F%2Fx%2Fa.h5',
) {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <DatasetComparisonPage />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  globalThis.localStorage.clear();
  sourceData.clear();

  vi.mocked(openPoseTraceSource).mockImplementation(
    async (file: H5File): Promise<PoseTraceSource> => {
      const data = sourceData.get(file.url);
      if (!data) {
        throw new Error(`No mock data for ${file.url}`);
      }

      return {
        sourceId: file.url,
        datasetName: file.name,
        demos: data.demos.map(demo),
        articulation: null,
        cleanup: vi.fn(),
      };
    },
  );

  vi.mocked(getDatasetProcessingInfo).mockImplementation(
    async (source: PoseTraceSource): Promise<DatasetProcessingSourceInfo> => {
      const data = sourceData.get(source.sourceId);
      if (!data) {
        throw new Error(`No mock info for ${source.sourceId}`);
      }

      return {
        keyPaths: data.keys.map((path) => ({
          path,
          availableInDemoCount: source.demos.length,
        })),
      };
    },
  );

  vi.mocked(loadDatasetComparisonValues).mockImplementation(
    async (
      source: PoseTraceSource,
      demoName: string,
      keyPaths: string[],
    ): Promise<DatasetComparisonValuesResult> => {
      const data = sourceData.get(source.sourceId);
      if (!data) {
        throw new Error(`No mock values for ${source.sourceId}`);
      }

      return {
        demoName,
        values: keyPaths.map((keyPath) => ({
          keyPath,
          status: 'ok',
          shape: [2, 2],
          dtype: 'float64',
          selection: ':,:',
          value: data.values[`${demoName}:${keyPath}`] ?? data.values[keyPath],
        })),
      };
    },
  );
});

afterEach(() => {
  useStore.setState(initialState, true);
  vi.clearAllMocks();
});

describe('DatasetComparisonPage', () => {
  it('renders common demo keys and raw values for two opened datasets', async () => {
    useStore.setState({
      opened: [
        makeRemote('https://x/a.h5', 'a.h5'),
        makeRemote('https://x/b.h5', 'b.h5'),
      ],
    });
    sourceData.set('https://x/a.h5', {
      demos: ['demo_0', 'demo_1'],
      keys: ['obs/pos', 'actions'],
      values: {
        'obs/pos': [
          [1, 2],
          [3, 4],
        ],
      },
    });
    sourceData.set('https://x/b.h5', {
      demos: ['demo_0'],
      keys: ['obs/pos', 'obs/vel'],
      values: {
        'obs/pos': [
          [5, 6],
          [7, 8],
        ],
      },
    });

    renderPage();

    await waitFor(() => {
      expect(loadDatasetComparisonValues).toHaveBeenCalledTimes(2);
    });

    const leftDemoSelect =
      screen.getByLabelText<HTMLSelectElement>('Left Demo');
    const rightDemoSelect =
      screen.getByLabelText<HTMLSelectElement>('Right Demo');
    expect([...leftDemoSelect.options].map((option) => option.value)).toEqual([
      'demo_0',
      'demo_1',
    ]);
    expect([...rightDemoSelect.options].map((option) => option.value)).toEqual([
      'demo_0',
    ]);
    expect(screen.getByText('obs/pos')).toBeInTheDocument();
    expect(screen.queryByText('actions')).not.toBeInTheDocument();
    expect(screen.queryByText('vel')).not.toBeInTheDocument();
    expect(screen.getAllByText('a.h5').length).toBeGreaterThan(0);
    expect(screen.getAllByText('b.h5').length).toBeGreaterThan(0);
    expect(screen.getByText('8')).toBeInTheDocument();
  });

  it('renders controls when one file is opened', async () => {
    useStore.setState({
      opened: [makeRemote('https://x/a.h5', 'a.h5')],
    });
    sourceData.set('https://x/a.h5', {
      demos: ['demo_0'],
      keys: ['obs/pos'],
      values: { 'obs/pos': [[1]] },
    });

    renderPage();

    await expect(
      screen.findByLabelText('Left Dataset'),
    ).resolves.toBeInTheDocument();
    expect(screen.getByLabelText('Right Dataset')).toBeInTheDocument();
  });

  it('allows comparing two demos from the same dataset', async () => {
    const user = userEvent.setup();
    useStore.setState({
      opened: [makeRemote('https://x/a.h5', 'a.h5')],
    });
    sourceData.set('https://x/a.h5', {
      demos: ['demo_0', 'demo_1'],
      keys: ['obs/pos'],
      values: {
        'demo_0:obs/pos': [[1]],
        'demo_1:obs/pos': [[2]],
      },
    });

    renderPage();

    await waitFor(() => {
      expect(loadDatasetComparisonValues).toHaveBeenCalledTimes(2);
    });

    await user.selectOptions(screen.getByLabelText('Right Demo'), 'demo_1');

    await waitFor(() => {
      expect(loadDatasetComparisonValues).toHaveBeenLastCalledWith(
        expect.objectContaining({ sourceId: 'https://x/a.h5' }),
        'demo_1',
        ['obs/pos'],
      );
    });
    expect(screen.getAllByText('1').length).toBeGreaterThan(0);
    expect(screen.getAllByText('2').length).toBeGreaterThan(0);
  });
});
