import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import Dropzone from './Dropzone';
import { FileService, useStore } from './stores';

const initialState = useStore.getState();

beforeEach(() => {
  globalThis.localStorage.clear();
  vi.spyOn(URL, 'createObjectURL').mockReturnValue('blob:mock');
});

afterEach(() => {
  useStore.setState(initialState, true);
});

function renderDropzone() {
  return render(
    <MemoryRouter>
      <Dropzone>
        <div data-testid="child">child</div>
      </Dropzone>
    </MemoryRouter>,
  );
}

// react-dropzone does not expose its file input via role/label, so a direct
// DOM query is the only way to reach it from tests.
function findFileInput(): HTMLInputElement | null {
  return document.querySelector<HTMLInputElement>('input[type="file"]');
}

describe('Dropzone', () => {
  it('renders its children inside the drop region', () => {
    renderDropzone();
    expect(screen.getByTestId('child')).toBeInTheDocument();
  });

  it('does not show the "Drop it!" overlay by default', () => {
    renderDropzone();
    expect(screen.queryByText('Drop it!')).not.toBeInTheDocument();
  });

  it('adds files chosen via the hidden input to the store', async () => {
    renderDropzone();
    const input = findFileInput();
    // eslint-disable-next-line vitest/no-conditional-in-test -- runtime narrow needed because querySelector is typed as nullable.
    if (!input) {
      throw new Error('Dropzone did not render its file input.');
    }

    const file = new File(['contents'], 'demo.h5', {
      type: 'application/x-hdf5',
    });
    await userEvent.upload(input, file);

    await vi.waitFor(() => {
      const { opened } = useStore.getState();
      expect(opened).toHaveLength(1);
      expect(opened[0]?.name).toBe('demo.h5');
      expect(opened[0]?.service).toBe(FileService.Local);
    });
  });
});
