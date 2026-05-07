import { render, screen } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { FileService, type H5File, useStore } from '../stores';
import Sidebar from './Sidebar';

const initialState = useStore.getState();

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
});

afterEach(() => {
  useStore.setState(initialState, true);
});

describe('Sidebar', () => {
  it('renders the brand link', () => {
    renderSidebar();
    expect(screen.getByRole('link', { name: /rebelHDF5/i })).toBeInTheDocument();
  });

  it('disables file-bound nav items when no file is open', () => {
    renderSidebar();
    expect(screen.getByRole('button', { name: 'Viewer' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Pose Trace' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Video Converter' })).toBeDisabled();
    expect(
      screen.getByRole('button', { name: 'Dataset Processing' }),
    ).toBeDisabled();
    expect(
      screen.getByRole('button', { name: 'Cloth Distribution' }),
    ).toBeDisabled();
  });

  it('enables file-bound nav items as links once a file is open', () => {
    useStore.setState({ opened: [makeRemote('https://x/a.h5', 'a.h5')] });
    renderSidebar('/view?url=https%3A%2F%2Fx%2Fa.h5');

    expect(screen.getByRole('link', { name: 'Viewer' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'Pose Trace' })).toBeInTheDocument();
    expect(
      screen.getByRole('link', { name: 'Video Converter' }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('link', { name: 'Dataset Processing' }),
    ).toBeInTheDocument();
  });

  it('always shows the Databricks nav link', () => {
    renderSidebar();
    expect(screen.getByRole('link', { name: 'Databricks' })).toBeInTheDocument();
  });

  it('disables the SidebarToggle when not viewing a file', () => {
    renderSidebar('/');
    expect(
      screen.getByRole('button', { name: /sidebar/i }),
    ).toBeDisabled();
  });

  it('shows the OpenedFiles flyout shell when collapsed and viewing a file', () => {
    useStore.setState({
      opened: [makeRemote('https://x/a.h5', 'a.h5')],
      sidebarMayCollapse: true,
    });
    renderSidebar('/view?url=https%3A%2F%2Fx%2Fa.h5');

    expect(
      screen.getByRole('button', { name: 'Opened files' }),
    ).toBeInTheDocument();
  });
});
