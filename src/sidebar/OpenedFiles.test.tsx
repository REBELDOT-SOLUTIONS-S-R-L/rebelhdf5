import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

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

function renderAt(initialEntry: string) {
  return render(
    <MemoryRouter initialEntries={[initialEntry]}>
      <OpenedFiles />
    </MemoryRouter>,
  );
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
      opened: [makeRemote('https://x/a.h5', 'a.h5'), makeRemote('https://x/b.h5', 'b.h5')],
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
      opened: [makeRemote('https://x/a.h5', 'a.h5'), makeRemote('https://x/b.h5', 'b.h5')],
    });

    renderAt('/');
    const removeButtons = screen.getAllByRole('button', { name: 'Remove file' });
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
    expect(link.getAttribute('href')).toContain(encodeURIComponent('https://x/a.h5'));
  });
});
