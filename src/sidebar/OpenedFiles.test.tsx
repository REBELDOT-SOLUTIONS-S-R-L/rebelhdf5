import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import {
  type AvailabilityState,
  type FileFeatureAvailability,
} from '../feature-availability';
import { FileService, type H5File, useStore } from '../stores';
import OpenedFiles from './OpenedFiles';

const initialState = useStore.getState();

function makeRemote(url: string, name = url): H5File {
  return {
    url,
    name,
    service: FileService.Url,
    resolvedUrl: url,
  };
}

function makeLocal(name = 'local.h5'): H5File {
  const file = new File([], name);
  const url = `blob:fake-${name}`;
  return {
    url,
    name,
    service: FileService.Local,
    resolvedUrl: url,
    file,
  };
}

function renderAt(initialEntry: string) {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <OpenedFiles availability={availabilityFor(useStore.getState().opened)} />
    </MemoryRouter>,
  );
}

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

function availabilityFor(opened: H5File[]): AvailabilityState {
  return {
    byUrl: Object.fromEntries(
      opened.map((file) => [file.url, availableFeatures()]),
    ),
    backendAvailable: true,
  };
}

beforeEach(() => {
  globalThis.localStorage.clear();
});

afterEach(() => {
  useStore.setState(initialState, true);
});

describe('OpenedFiles', () => {
  it('shows the empty hint when no file is opened', () => {
    renderAt('/');
    expect(screen.getByText(/please open a file/i)).toBeInTheDocument();
    expect(screen.queryByRole('list')).not.toBeInTheDocument();
  });

  it('renders one entry per opened file', () => {
    useStore.setState({
      opened: [
        makeRemote('https://x/a.h5', 'a.h5'),
        makeRemote('https://x/b.h5', 'b.h5'),
      ],
    });

    renderAt('/view?url=https%3A%2F%2Fx%2Fa.h5');

    const items = screen.getAllByRole('listitem');
    expect(items).toHaveLength(2);
    expect(within(items[0]).getByText('a.h5')).toBeInTheDocument();
    expect(within(items[1]).getByText('b.h5')).toBeInTheDocument();
  });

  it('marks the active file with aria-current', () => {
    const active = makeRemote('https://x/a.h5', 'a.h5');
    const other = makeRemote('https://x/b.h5', 'b.h5');
    useStore.setState({ opened: [active, other] });

    renderAt('/view?url=https%3A%2F%2Fx%2Fa.h5');

    const links = screen.getAllByRole('link', { name: /\.h5$/ });
    expect(links[0]).toHaveAttribute('aria-current', 'page');
    expect(links[1]).not.toHaveAttribute('aria-current');
  });

  it('removes a file from the store when its trash button is clicked', async () => {
    useStore.setState({
      opened: [
        makeRemote('https://x/a.h5', 'a.h5'),
        makeRemote('https://x/b.h5', 'b.h5'),
      ],
    });

    renderAt('/');
    const removeButtons = screen.getAllByRole('button', {
      name: 'Remove file',
    });
    await userEvent.click(removeButtons[0]);

    expect(useStore.getState().opened.map((f) => f.url)).toEqual([
      'https://x/b.h5',
    ]);
  });

  it('routes the entry link to the current feature route', () => {
    useStore.setState({ opened: [makeRemote('https://x/a.h5', 'a.h5')] });
    renderAt('/pose-trace?url=https%3A%2F%2Fx%2Fa.h5');

    const link = screen.getByRole('link', { name: 'a.h5' });
    expect(link.getAttribute('href')).toContain('/pose-trace?');
    expect(link.getAttribute('href')).toContain(
      encodeURIComponent('https://x/a.h5'),
    );
  });

  it('routes entry links to the dataset comparison page when active', () => {
    useStore.setState({ opened: [makeRemote('https://x/a.h5', 'a.h5')] });
    renderAt('/dataset-comparison?url=https%3A%2F%2Fx%2Fa.h5');

    const link = screen.getByRole('link', { name: 'a.h5' });
    expect(link.getAttribute('href')).toContain('/dataset-comparison?');
    expect(link.getAttribute('href')).toContain(
      encodeURIComponent('https://x/a.h5'),
    );
  });

  it('disables entry navigation when the current route is unavailable', () => {
    const file = makeRemote('https://x/a.h5', 'a.h5');
    const unavailableFeatures = availableFeatures();
    unavailableFeatures.poseTrace = {
      status: 'unavailable',
      reason: 'No usable pose data found.',
    };
    useStore.setState({ opened: [file] });

    render(
      <MemoryRouter initialEntries={['/pose-trace?url=https%3A%2F%2Fx%2Fa.h5']}>
        <OpenedFiles
          availability={{
            byUrl: { [file.url]: unavailableFeatures },
            backendAvailable: true,
          }}
        />
      </MemoryRouter>,
    );

    expect(screen.getByRole('button', { name: 'a.h5' })).toBeDisabled();
    expect(screen.queryByRole('link', { name: 'a.h5' })).toBeNull();
  });

  it('keeps dataset comparison row navigation enabled for unsupported schemas', () => {
    const file = makeRemote('https://x/a.h5', 'a.h5');
    const unavailableFeatures = availableFeatures();
    unavailableFeatures.datasetComparison = {
      status: 'unavailable',
      reason: 'Unsupported HDF5 schema.',
    };
    useStore.setState({ opened: [file] });

    render(
      <MemoryRouter
        initialEntries={['/dataset-comparison?url=https%3A%2F%2Fx%2Fa.h5']}
      >
        <OpenedFiles
          availability={{
            byUrl: { [file.url]: unavailableFeatures },
            backendAvailable: true,
          }}
        />
      </MemoryRouter>,
    );

    expect(screen.getByRole('link', { name: 'a.h5' })).toBeInTheDocument();
  });

  it('keeps remote files downloadable', () => {
    useStore.setState({
      opened: [makeRemote('https://x/a.h5', 'a.h5')],
    });

    renderAt('/view?url=https%3A%2F%2Fx%2Fa.h5');

    const downloadLink = screen.getByRole('link', { name: 'Download file' });
    expect(downloadLink).toHaveAttribute('href', 'https://x/a.h5');
    expect(downloadLink).toHaveAttribute('download', 'a.h5');
  });

  it('disables downloads for local files', () => {
    const local = makeLocal('local.h5');
    useStore.setState({ opened: [local] });

    renderAt(`/view?url=${encodeURIComponent(local.url)}`);

    expect(
      screen.getByRole('button', { name: 'Download file' }),
    ).toBeDisabled();
    expect(
      screen.queryByRole('link', { name: 'Download file' }),
    ).not.toBeInTheDocument();
  });
});
