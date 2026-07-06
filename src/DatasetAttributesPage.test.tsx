import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import DatasetAttributesPage from './DatasetAttributesPage';
import { FileService, type H5File, useStore } from './stores';

const mocks = vi.hoisted(() => ({
  checkBackend: vi.fn(),
  getDatasetAttributes: vi.fn(),
  updateDatasetArticulation: vi.fn(),
}));

vi.mock('./python-backend', () => ({
  checkBackend: mocks.checkBackend,
  getDatasetAttributes: mocks.getDatasetAttributes,
  updateDatasetArticulation: mocks.updateDatasetArticulation,
}));

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

const attributesResult = {
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

function renderPage(): void {
  render(
    <MemoryRouter initialEntries={['/dataset-attributes']}>
      <DatasetAttributesPage />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  useStore.setState({ opened: [] }, false);
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
    useStore.setState({ opened: [localFile('/data/attrs.hdf5')] }, false);
    renderPage();

    expect(
      await screen.findByText(/Python backend is not available/u),
    ).toBeInTheDocument();
    expect(mocks.getDatasetAttributes).not.toHaveBeenCalled();
  });

  it('loads attributes and renders the articulation editor', async () => {
    useStore.setState({ opened: [localFile('/data/attrs.hdf5')] }, false);
    renderPage();

    // Group attribute panel renders with its slash-nested title.
    expect(await screen.findByText('data.attrs')).toBeInTheDocument();
    expect(mocks.getDatasetAttributes).toHaveBeenCalledWith('/data/attrs.hdf5');

    // Articulation fields are populated from the backend payload.
    expect(screen.getByDisplayValue('robot')).toBeInTheDocument();
    expect(screen.getByDisplayValue('7')).toBeInTheDocument();
    expect(screen.getByText('Source: attribute')).toBeInTheDocument();
  });

  it('saves articulation edits via updateDatasetArticulation', async () => {
    const user = userEvent.setup();
    useStore.setState({ opened: [localFile('/data/attrs.hdf5')] }, false);
    renderPage();

    await screen.findByDisplayValue('robot');
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
    expect(
      await screen.findByText('Articulation attributes saved.'),
    ).toBeInTheDocument();
  });
});
