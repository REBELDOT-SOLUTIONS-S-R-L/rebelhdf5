import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import DatasetAttributesPage from './DatasetAttributesPage';
import {
  type AvailabilityState,
  type FileFeatureAvailability,
} from './feature-availability';
import {
  type checkBackend,
  type DatasetAttributesResult,
  type getDatasetAttributes,
  type updateDatasetArticulation,
} from './python-backend';
import { FileService, type H5File, useStore } from './stores';

const mocks = vi.hoisted(() => ({
  checkBackend: vi.fn<typeof checkBackend>(),
  getDatasetAttributes: vi.fn<typeof getDatasetAttributes>(),
  updateDatasetArticulation: vi.fn<typeof updateDatasetArticulation>(),
}));
let availabilityState: AvailabilityState = {
  byUrl: {},
  backendAvailable: true,
};

vi.mock(import('./python-backend'), () => ({
  checkBackend: mocks.checkBackend,
  getDatasetAttributes: mocks.getDatasetAttributes,
  updateDatasetArticulation: mocks.updateDatasetArticulation,
}));

vi.mock(import('./feature-availability'), async (importOriginal) => {
  const actual = await importOriginal();
  return {
    ...actual,
    useDatasetFeatureAvailability: () => availabilityState,
  };
});

function localFile(serverPath?: string): H5File {
  return {
    url: 'u-attrs.hdf5',
    name: 'attrs.hdf5',
    service: FileService.Local,
    resolvedUrl: 'blob:attrs.hdf5',
    serverPath,
    file: new File([new Uint8Array([1])], 'attrs.hdf5'),
  };
}

const attributesResult: DatasetAttributesResult = {
  path: '/data/attrs.hdf5',
  attrs: { total: 12 },
  articulation: {
    name: 'robot',
    joint_number: 7,
    segmentation: { arm: { target: '[0:7]', obs: '[0:7]' } },
    end_effectors: {},
  },
  articulationSource: 'attribute',
  groups: [
    { path: '/data', attrs: { total: 12, 'articulation/name': 'robot' } },
  ],
};

function availableFeatures(): FileFeatureAvailability {
  const available = { status: 'available', reason: '' } as const;
  return {
    poseTrace: available,
    videoConverter: available,
    datasetProcessing: available,
    datasetComparison: available,
    objectDistribution: available,
  };
}

function unavailableFeatures(): FileFeatureAvailability {
  const unavailable = {
    status: 'unavailable',
    reason: 'Unsupported HDF5 schema.',
  } as const;
  return {
    poseTrace: unavailable,
    videoConverter: unavailable,
    datasetProcessing: unavailable,
    datasetComparison: unavailable,
    objectDistribution: unavailable,
  };
}

function setAvailability(file: H5File, features: FileFeatureAvailability) {
  availabilityState = {
    byUrl: { [file.url]: features },
    backendAvailable: true,
  };
}

function renderPage(): void {
  render(
    <MemoryRouter initialEntries={['/dataset-attributes']}>
      <DatasetAttributesPage />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  useStore.setState({ opened: [] }, false);
  availabilityState = { byUrl: {}, backendAvailable: true };
  mocks.checkBackend.mockResolvedValue({
    available: true,
    rootDir: '/data',
    version: 3,
  });
  mocks.getDatasetAttributes.mockResolvedValue(attributesResult);
  mocks.updateDatasetArticulation.mockResolvedValue(attributesResult);
});

afterEach(() => {
  useStore.setState({ opened: [] }, false);
});

describe('DatasetAttributesPage', () => {
  it('prompts to open a file when none is available', () => {
    renderPage();
    expect(
      screen.getByText('Open an HDF5 file before editing dataset attributes.'),
    ).toBeInTheDocument();
  });

  it('shows a backend-unavailable message when the backend is down', async () => {
    mocks.checkBackend.mockResolvedValue({
      available: false,
      rootDir: null,
      version: null,
    });
    const file = localFile('/data/attrs.hdf5');
    setAvailability(file, availableFeatures());
    useStore.setState({ opened: [file] }, false);
    renderPage();

    await expect(
      screen.findByText(/Python backend is not available/u),
    ).resolves.toBeInTheDocument();
    expect(mocks.getDatasetAttributes).not.toHaveBeenCalled();
  });

  it('loads attributes and renders the articulation editor', async () => {
    const file = localFile('/data/attrs.hdf5');
    setAvailability(file, availableFeatures());
    useStore.setState({ opened: [file] }, false);
    renderPage();

    // Group attribute panel renders with its slash-nested title.
    await expect(screen.findByText('data.attrs')).resolves.toBeInTheDocument();
    expect(mocks.getDatasetAttributes).toHaveBeenCalledWith('/data/attrs.hdf5');

    // Articulation fields are populated from the backend payload.
    expect(screen.getByDisplayValue('robot')).toBeInTheDocument();
    expect(screen.getByDisplayValue('7')).toBeInTheDocument();
    expect(screen.getByText('Source: attribute')).toBeInTheDocument();
  });

  it('saves articulation edits via updateDatasetArticulation', async () => {
    const user = userEvent.setup();
    const file = localFile('/data/attrs.hdf5');
    setAvailability(file, availableFeatures());
    useStore.setState({ opened: [file] }, false);
    renderPage();

    await expect(
      screen.findByDisplayValue('robot'),
    ).resolves.toBeInTheDocument();
    const [saveButton] = screen.getAllByRole('button', {
      name: 'Save Articulation',
    });
    await user.click(saveButton);

    await waitFor(() => {
      expect(mocks.updateDatasetArticulation).toHaveBeenCalledWith(
        '/data/attrs.hdf5',
        {
          name: 'robot',
          joint_number: 7,
          segmentation: { arm: { target: '[0:7]', obs: '[0:7]' } },
          end_effectors: {},
        },
      );
    });
    await expect(
      screen.findByText('Articulation attributes saved.'),
    ).resolves.toBeInTheDocument();
  });

  it('does not load attributes for unsupported schemas', async () => {
    const file = localFile('/data/dummy.hdf5');
    setAvailability(file, unavailableFeatures());
    useStore.setState({ opened: [file] }, false);
    renderPage();

    await expect(
      screen.findByText('Dataset attributes require a standard demo dataset.'),
    ).resolves.toBeInTheDocument();
    expect(mocks.getDatasetAttributes).not.toHaveBeenCalled();
  });
});
