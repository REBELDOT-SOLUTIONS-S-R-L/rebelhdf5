import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { describe, expect, it } from 'vitest';

import HttpErrorMessage from './HttpErrorMessage';

function renderMsg(status: number, fileUrl = 'https://example.com/file.h5') {
  return render(
    <MemoryRouter>
      <HttpErrorMessage status={status} fileUrl={fileUrl} />
    </MemoryRouter>,
  );
}

describe('HttpErrorMessage', () => {
  it('renders nothing for unhandled status codes', () => {
    const { container } = renderMsg(500);
    expect(container).toBeEmptyDOMElement();
  });

  it('explains 400 as a malformed URL', () => {
    renderMsg(400);
    expect(
      screen.getByText(/URL of the file may be wrong or incomplete/i),
    ).toBeInTheDocument();
  });

  it('explains 401 as an auth-required file', () => {
    renderMsg(401);
    expect(
      screen.getByText(/Authentication is required/i),
    ).toBeInTheDocument();
  });

  it('suggests permalinks for 404 on an unstable ref', () => {
    renderMsg(404, 'https://github.com/foo/bar/blob/main/file.h5');
    expect(
      screen.getByText(/repository URL pointing to a development branch/i),
    ).toBeInTheDocument();
  });

  it('explains 404 generically for stable URLs', () => {
    renderMsg(404, 'https://example.com/file.h5');
    expect(
      screen.getByText(/may no longer exist at this URL/i),
    ).toBeInTheDocument();
  });

  it('offers the teabag fallback for 418', () => {
    renderMsg(418);
    expect(
      screen.getByRole('link', { name: 'this file' }),
    ).toBeInTheDocument();
  });
});
