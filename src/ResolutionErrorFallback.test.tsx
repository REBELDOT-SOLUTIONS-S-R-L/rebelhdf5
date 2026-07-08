import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { NetworkError } from './fetch-utils';
import ResolutionErrorFallback from './ResolutionErrorFallback';

function renderFallback(
  error: unknown,
  fileUrl = 'https://example.com/file.h5',
) {
  return render(
    <ResolutionErrorFallback
      error={error}
      fileUrl={fileUrl}
      resetErrorBoundary={() => {}}
    />,
  );
}

describe('ResolutionErrorFallback', () => {
  it('renders the message and an "Unknown error" placeholder for non-Error values', () => {
    renderFallback('not-an-error-object');
    expect(screen.getByText('Unknown error')).toBeInTheDocument();
  });

  it('renders the error message verbatim for Error instances', () => {
    renderFallback(new Error('cannot resolve URL'));
    expect(screen.getByText('cannot resolve URL')).toBeInTheDocument();
  });

  it('adds the network-error hint when error is a NetworkError', () => {
    renderFallback(new NetworkError());
    expect(screen.getByText(/cross-origin request/i)).toBeInTheDocument();
  });

  it('exposes the offending URL inside the debug details', () => {
    renderFallback(new Error('bad'), 'https://oops.example.com/x.h5');
    expect(
      screen.getByRole('link', { name: 'https://oops.example.com/x.h5' }),
    ).toBeInTheDocument();
  });
});
