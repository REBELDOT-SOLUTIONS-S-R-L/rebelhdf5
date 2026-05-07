import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import LocalService from './LocalService';

const openFilePicker = vi.fn();

vi.mock('../Dropzone', () => ({
  useDropzoneContext: () => ({ openFilePicker }),
}));

describe('LocalService', () => {
  it('renders the picker button and h5wasm credit link', () => {
    render(<LocalService />);
    expect(
      screen.getByRole('button', { name: 'Select HDF5 files' }),
    ).toBeInTheDocument();
    expect(screen.getByRole('link', { name: 'h5wasm' })).toHaveAttribute(
      'href',
      'https://github.com/usnistgov/h5wasm',
    );
  });

  it('calls openFilePicker from the dropzone context on click', async () => {
    render(<LocalService />);
    await userEvent.click(
      screen.getByRole('button', { name: 'Select HDF5 files' }),
    );
    expect(openFilePicker).toHaveBeenCalledTimes(1);
  });
});
