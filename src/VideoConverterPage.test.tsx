import { render, screen, waitFor, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  type listDemoVideos,
  type loadDemoVideoFrames,
  type openPoseTraceSource,
} from './pose-trace/hdf5';
import { FileService, type H5File, useStore } from './stores';
import VideoConverterPage from './VideoConverterPage';

const mocks = vi.hoisted(() => ({
  listDemoVideos: vi.fn<typeof listDemoVideos>(),
  loadDemoVideoFrames: vi.fn<typeof loadDemoVideoFrames>(),
  openPoseTraceSource: vi.fn<typeof openPoseTraceSource>(),
}));

vi.mock(import('./pose-trace/hdf5'), () => ({
  listDemoVideos: mocks.listDemoVideos,
  loadDemoVideoFrames: mocks.loadDemoVideoFrames,
  openPoseTraceSource: mocks.openPoseTraceSource,
}));

function localFile(name = 'clip.hdf5'): H5File {
  return {
    url: `u-${name}`,
    name,
    service: FileService.Local,
    resolvedUrl: `blob:${name}`,
    file: new File([new Uint8Array([1])], name),
  };
}

function renderPage(initialUrl = '/video-converter'): void {
  render(
    <MemoryRouter initialEntries={[initialUrl]}>
      <VideoConverterPage />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  useStore.setState({ opened: [] }, false);
  mocks.openPoseTraceSource.mockResolvedValue({
    sourceId: 's1',
    datasetName: 'clip',
    demos: [
      {
        name: 'demo_0',
        episode_index: 0,
        num_samples: 5,
        success: true,
        source_episode_index: 0,
      },
    ],
    articulation: null,
    cleanup: vi.fn<() => void>(),
  });
  mocks.listDemoVideos.mockResolvedValue([]);
});

afterEach(() => {
  useStore.setState({ opened: [] }, false);
});

describe('VideoConverterPage', () => {
  it('shows the empty state when no file is open', () => {
    renderPage();
    expect(
      screen.getByText(/Open an HDF5 file in rebelHDF5/u),
    ).toBeInTheDocument();
  });

  it('opens the source and renders the demo selector for the resolved file', async () => {
    const file = localFile('clip.hdf5');
    useStore.setState({ opened: [file] }, false);
    renderPage(`/video-converter?url=${encodeURIComponent(file.url)}`);

    // The demo dropdown appears once the (mocked) source is opened.
    const demoSelect = await screen.findByLabelText('Demo');
    expect(mocks.openPoseTraceSource).toHaveBeenCalledWith(file);
    expect(
      within(demoSelect).getByRole('option', { name: /demo_0/u }),
    ).toBeInTheDocument();

    // Status row reflects the single demo.
    await waitFor(() => {
      expect(screen.getByLabelText('Demo count')).toHaveTextContent('1');
    });
    // With no videos, the video selector shows the empty message.
    await expect(
      screen.findByText('No supported videos found'),
    ).resolves.toBeInTheDocument();
  });
});
