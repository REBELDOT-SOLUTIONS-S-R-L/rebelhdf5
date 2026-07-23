import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import Dropzone, { useDropzoneContext } from './Dropzone';
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

function getFileInput(): HTMLInputElement {
  return screen.getByLabelText<HTMLInputElement>('HDF5 file input');
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
    const input = getFileInput();

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

function PickerProbe() {
  const { openFilePicker } = useDropzoneContext();
  return (
    <button
      type="button"
      onClick={() => {
        void openFilePicker();
      }}
    >
      Pick
    </button>
  );
}

function renderWithPicker() {
  return render(
    <MemoryRouter>
      <Dropzone>
        <PickerProbe />
      </Dropzone>
    </MemoryRouter>,
  );
}

describe('Dropzone openFilePicker', () => {
  afterEach(() => {
    Reflect.deleteProperty(globalThis, 'rebelHdf5Desktop');
    Reflect.deleteProperty(globalThis, 'showOpenFilePicker');
  });

  it('falls back to the hidden input when no native picker is available', async () => {
    renderWithPicker();
    const input = getFileInput();
    const inputClick = vi.spyOn(input, 'click');

    await userEvent.click(screen.getByRole('button', { name: 'Pick' }));
    expect(inputClick).toHaveBeenCalledOnce();
  });

  it('uses the desktop launcher when window.rebelHdf5Desktop is present', async () => {
    Object.assign(globalThis, {
      rebelHdf5Desktop: {
        getPathForFile:
          vi.fn<NonNullable<RebelHdf5DesktopRuntime['getPathForFile']>>(),
      },
    });

    renderWithPicker();
    const input = getFileInput();
    const inputClick = vi.spyOn(input, 'click');

    await userEvent.click(screen.getByRole('button', { name: 'Pick' }));
    expect(inputClick).toHaveBeenCalledOnce();
  });

  it('opens via showOpenFilePicker when supported and adds returned files', async () => {
    const file = new File(['x'], 'fsa.h5', { type: 'application/x-hdf5' });
    const handle = {
      getFile: vi.fn<FileSystemFileHandle['getFile']>().mockResolvedValue(file),
    };
    Object.assign(globalThis, {
      showOpenFilePicker: vi
        .fn<NonNullable<typeof globalThis.showOpenFilePicker>>()
        .mockResolvedValue([handle as unknown as FileSystemFileHandle]),
    });

    renderWithPicker();
    await userEvent.click(screen.getByRole('button', { name: 'Pick' }));

    await vi.waitFor(() => {
      const { opened } = useStore.getState();
      expect(opened.map((f) => f.name)).toEqual(['fsa.h5']);
    });
  });

  it('silently returns when the user aborts showOpenFilePicker', async () => {
    Object.assign(globalThis, {
      showOpenFilePicker: vi
        .fn<NonNullable<typeof globalThis.showOpenFilePicker>>()
        .mockRejectedValue(new DOMException('User cancelled', 'AbortError')),
    });

    renderWithPicker();
    await userEvent.click(screen.getByRole('button', { name: 'Pick' }));

    expect(useStore.getState().opened).toHaveLength(0);
  });

  it('falls through to the hidden input when showOpenFilePicker fails for any other reason', async () => {
    Object.assign(globalThis, {
      showOpenFilePicker: vi
        .fn<NonNullable<typeof globalThis.showOpenFilePicker>>()
        .mockRejectedValue(new Error('boom')),
    });

    renderWithPicker();
    const input = getFileInput();
    const inputClick = vi.spyOn(input, 'click');

    await userEvent.click(screen.getByRole('button', { name: 'Pick' }));
    expect(inputClick).toHaveBeenCalledOnce();
  });
});
