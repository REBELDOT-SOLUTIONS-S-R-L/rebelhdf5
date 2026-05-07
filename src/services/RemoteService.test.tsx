import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { describe, expect, it } from 'vitest';

import RemoteService from './RemoteService';

function LocationProbe() {
  const location = useLocation();
  return (
    <div data-testid="location">
      {location.pathname}
      {location.search}
    </div>
  );
}

function renderRemote() {
  return render(
    <MemoryRouter>
      <Routes>
        <Route
          path="*"
          element={
            <>
              <RemoteService />
              <LocationProbe />
            </>
          }
        />
      </Routes>
    </MemoryRouter>,
  );
}

describe('RemoteService', () => {
  it('shows the default hint until interaction', () => {
    renderRemote();
    expect(
      screen.getByText(
        /Paste the URL of a file from a Zenodo record or GitHub repository/i,
      ),
    ).toBeInTheDocument();
  });

  it('navigates to /view with the URL when a valid URL is submitted', async () => {
    renderRemote();

    await userEvent.type(
      screen.getByRole('textbox', { name: 'URL of HDF5 file' }),
      'https://example.com/file.h5',
    );
    await userEvent.click(screen.getByRole('button', { name: 'Open' }));

    expect(screen.getByTestId('location').textContent).toContain('/view');
    expect(screen.getByTestId('location').textContent).toContain(
      encodeURIComponent('https://example.com/file.h5'),
    );
  });

  it('shows a validation error and does not navigate on empty submit', async () => {
    renderRemote();
    await userEvent.click(screen.getByRole('button', { name: 'Open' }));

    expect(screen.getByRole('textbox')).toHaveAttribute('data-error', 'true');
    expect(screen.getByTestId('location').textContent).toBe('/');
  });

  it('warns about unstable refs in the URL', async () => {
    renderRemote();
    await userEvent.type(
      screen.getByRole('textbox', { name: 'URL of HDF5 file' }),
      'https://github.com/foo/bar/blob/main/file.h5',
    );

    expect(screen.getByRole('link', { name: 'permalink' })).toBeInTheDocument();
  });
});
