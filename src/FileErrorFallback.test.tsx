import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it, vi } from 'vitest';

import { FetchError, NetworkError } from './fetch-utils';
import FileErrorFallback from './FileErrorFallback';
import { FileService, type H5File, type RemoteFile } from './stores';

function makeFile(overrides: Partial<RemoteFile> = {}): RemoteFile {
  return {
    url: 'https://example.com/file.h5',
    name: 'file.h5',
    service: FileService.Url,
    resolvedUrl: 'https://example.com/file.h5',
    ...overrides,
  };
}

function renderFallback(props: {
  error: unknown;
  file?: H5File;
  resetErrorBoundary?: () => void;
}) {
  const file = props.file ?? makeFile();
  return render(
    <MemoryRouter>
      <FileErrorFallback
        error={props.error}
        file={file}
        resetErrorBoundary={props.resetErrorBoundary ?? (() => {})}
      />
    </MemoryRouter>,
  );
}

describe('FileErrorFallback', () => {
  it('renders the error message and a Retry button', () => {
    renderFallback({ error: new Error('boom') });
    expect(screen.getByText('boom')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument();
  });

  it('falls back to "Unknown error" when error is not an Error instance', () => {
    renderFallback({ error: 'something raw' });
    expect(screen.getByText('Unknown error')).toBeInTheDocument();
  });

  it('opens a <details> with the inner cause when present', () => {
    const cause = new Error('underlying');
    const error = new Error('outer', { cause });
    renderFallback({ error });

    expect(screen.getByText('outer')).toBeInTheDocument();
    expect(screen.getByText('underlying')).toBeInTheDocument();
  });

  it('shows the network-error CTA and download link when fetch fails', () => {
    renderFallback({ error: new NetworkError() });
    expect(screen.getByText(/file could not be fetched/i)).toBeInTheDocument();
    expect(
      screen.getByRole('link', { name: 'Download file' }),
    ).toBeInTheDocument();
  });

  it('embeds an HttpErrorMessage for FetchError instances', () => {
    renderFallback({ error: new FetchError(404, 'Not Found') });
    expect(
      screen.getByText(/may no longer exist at this url/i),
    ).toBeInTheDocument();
  });

  it('lists the resolved URL only when it differs from the file URL', () => {
    renderFallback({
      error: new Error('boom'),
      file: makeFile({
        url: 'https://example.com/a.h5',
        resolvedUrl: 'https://cdn.example.com/a.h5',
      }),
    });
    expect(screen.getByText(/resolved url:/i)).toBeInTheDocument();
  });

  it('calls resetErrorBoundary when Retry is clicked', async () => {
    const reset = vi.fn<() => void>();
    renderFallback({ error: new Error('boom'), resetErrorBoundary: reset });
    await userEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(reset).toHaveBeenCalledOnce();
  });
});
