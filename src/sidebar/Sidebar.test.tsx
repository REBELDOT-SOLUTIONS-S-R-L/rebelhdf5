import { render, screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  type DatasetFeature,
  type FeatureAvailability,
} from '../feature-availability';
import { FileService, type H5File, useStore } from '../stores';
import Sidebar from './Sidebar';

const initialState = useStore.getState();
let featureStatuses: Partial<Record<DatasetFeature, FeatureAvailability>> = {};

vi.mock(import('../feature-availability'), async (importOriginal) => {
  const actual = await importOriginal();
  const available: FeatureAvailability = {
    status: 'available',
    reason: '',
  };
  const unavailable: FeatureAvailability = {
    status: 'unavailable',
    reason: 'Unavailable in test',
  };

  return {
    ...actual,
    useDatasetFeatureAvailability: () => ({
      byUrl: {},
      backendAvailable: true,
    }),
    getDatasetFeatureAvailability: ({
      file,
      feature,
    }: {
      file: H5File | null;
      feature: DatasetFeature;
    }) => {
      if (!file) {
        return unavailable;
      }

      if (feature === 'datasetComparison') {
        return available;
      }

      return featureStatuses[feature] ?? available;
    },
  };
});

function makeRemote(url: string, name = url): H5File {
  return {
    url,
    name,
    service: FileService.Url,
    resolvedUrl: url,
  };
}

function renderSidebar(initialEntry = '/') {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <Routes>
        <Route path="*" element={<Sidebar />} />
      </Routes>
    </MemoryRouter>,
  );
}

beforeEach(() => {
  globalThis.localStorage.clear();
  featureStatuses = {};
});

afterEach(() => {
  useStore.setState(initialState, true);
});

describe('Sidebar', () => {
  it('renders the brand link', () => {
    renderSidebar();
    expect(
      screen.getByRole('link', { name: /rebelhdf5/i }),
    ).toBeInTheDocument();
  });

  it('disables file-bound nav items when no file is open', () => {
    renderSidebar();
    expect(screen.getByRole('button', { name: 'Viewer' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Pose Trace' })).toBeDisabled();
    expect(
      screen.getByRole('button', { name: 'Video Converter' }),
    ).toBeDisabled();
    expect(
      screen.getByRole('button', { name: 'Dataset Processing' }),
    ).toBeDisabled();
    expect(
      screen.getByRole('button', { name: 'Dataset Comparison' }),
    ).toBeDisabled();
    expect(
      screen.getByRole('button', { name: 'Object Distribution' }),
    ).toBeDisabled();
  });

  it('enables file-bound nav items as links once a file is open', () => {
    useStore.setState({ opened: [makeRemote('https://x/a.h5', 'a.h5')] });
    renderSidebar('/view?url=https%3A%2F%2Fx%2Fa.h5');

    expect(screen.getByRole('link', { name: 'Viewer' })).toBeInTheDocument();
    expect(
      screen.getByRole('link', { name: 'Pose Trace' }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('link', { name: 'Video Converter' }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('link', { name: 'Dataset Processing' }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('link', { name: 'Dataset Comparison' }),
    ).toBeInTheDocument();
  });

  it('disables file-bound nav items while availability is pending', () => {
    featureStatuses = {
      poseTrace: {
        status: 'pending',
        reason: 'Inspecting dataset schema...',
      },
    };
    useStore.setState({ opened: [makeRemote('https://x/a.h5', 'a.h5')] });
    renderSidebar('/view?url=https%3A%2F%2Fx%2Fa.h5');

    expect(screen.getByRole('button', { name: 'Pose Trace' })).toBeDisabled();
    expect(screen.queryByRole('link', { name: 'Pose Trace' })).toBeNull();
  });

  it('disables unavailable file-bound nav items', () => {
    featureStatuses = {
      poseTrace: {
        status: 'unavailable',
        reason: 'No usable pose data found.',
      },
    };
    useStore.setState({ opened: [makeRemote('https://x/a.h5', 'a.h5')] });
    renderSidebar('/view?url=https%3A%2F%2Fx%2Fa.h5');

    expect(screen.getByRole('button', { name: 'Pose Trace' })).toBeDisabled();
    expect(screen.queryByRole('link', { name: 'Pose Trace' })).toBeNull();
  });

  it('keeps Dataset Comparison enabled even when schema inspection fails', () => {
    featureStatuses = {
      datasetComparison: {
        status: 'unavailable',
        reason: 'Unsupported HDF5 schema.',
      },
    };
    useStore.setState({ opened: [makeRemote('https://x/a.h5', 'a.h5')] });
    renderSidebar('/view?url=https%3A%2F%2Fx%2Fa.h5');

    expect(
      screen.getByRole('link', { name: 'Dataset Comparison' }),
    ).toBeInTheDocument();
  });

  it('always shows the Databricks nav link', () => {
    renderSidebar();
    expect(
      screen.getByRole('link', { name: 'Databricks' }),
    ).toBeInTheDocument();
  });

  it('enables the SidebarToggle outside the viewer page', () => {
    renderSidebar('/');
    expect(screen.getByRole('button', { name: /sidebar/i })).toBeEnabled();
  });

  it('shows the OpenedFiles flyout shell when collapsed', () => {
    useStore.setState({
      opened: [makeRemote('https://x/a.h5', 'a.h5')],
      sidebarMayCollapse: true,
    });
    renderSidebar('/dataset-processing?url=https%3A%2F%2Fx%2Fa.h5');

    expect(
      screen.getByRole('button', { name: 'Opened files' }),
    ).toBeInTheDocument();
  });
});
