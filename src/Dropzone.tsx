import {
  createContext,
  type PropsWithChildren,
  useCallback,
  useContext,
} from 'react';
import { useDropzone } from 'react-dropzone';
import { useNavigate } from 'react-router-dom';

import styles from './Dropzone.module.css';
import { FileService, type LocalFile, useStore } from './stores';
import { getViewerLink } from './utils';

interface DropzoneContextValue {
  openFilePicker: () => void;
}

const DropzoneContext = createContext({} as DropzoneContextValue);

interface Props {}

interface PendingLocalFile {
  file: File;
  serverPath?: string;
}

const OPEN_PICKER_OPTIONS: OpenFilePickerOptions = {
  id: 'rebelhdf5-open-hdf5-files',
  multiple: true,
  types: [
    {
      description: 'HDF5 files',
      accept: {
        'application/x-hdf5': ['.h5', '.hdf5'],
      },
    },
  ],
};

function getDesktopFilePath(file: File): string | undefined {
  try {
    return globalThis.rebelHdf5Desktop?.getPathForFile?.(file) || undefined;
  } catch {
    return undefined;
  }
}

function Dropzone(props: PropsWithChildren<Props>) {
  const { children } = props;

  const openFiles = useStore((state) => state.openFiles);
  const navigate = useNavigate();

  const openLocalFiles = useCallback(
    (files: PendingLocalFile[]) => {
      if (files.length === 0) {
        return;
      }

      const h5Files = files.map<LocalFile>(({ file, serverPath }) => {
        const url = URL.createObjectURL(file);
        return {
          name: file.name,
          url,
          service: FileService.Local,
          resolvedUrl: url,
          file,
          ...(serverPath ? { serverPath } : {}),
        };
      });

      openFiles(h5Files);

      navigate(getViewerLink(h5Files[0].url));
    },
    [openFiles, navigate],
  );

  const onDropAccepted = useCallback(
    (files: File[]) => {
      openLocalFiles(
        files.map((file) => ({
          file,
          serverPath: getDesktopFilePath(file),
        })),
      );
    },
    [openLocalFiles],
  );

  const { getRootProps, getInputProps, open, isDragActive } = useDropzone({
    noClick: true,
    noKeyboard: true,
    onDropAccepted,
  });

  const openFilePicker = useCallback(async () => {
    if (globalThis.rebelHdf5Desktop) {
      open();
      return;
    }

    if (globalThis.showOpenFilePicker) {
      try {
        const handles = await globalThis.showOpenFilePicker(OPEN_PICKER_OPTIONS);
        const files = await Promise.all(
          handles.map(async (handle) => {
            const file = await handle.getFile();
            return {
              file,
              serverPath: getDesktopFilePath(file),
            };
          }),
        );
        openLocalFiles(files);
        return;
      } catch (error: unknown) {
        if (error instanceof DOMException && error.name === 'AbortError') {
          return;
        }
      }
    }

    open();
  }, [open, openLocalFiles]);

  return (
    <div {...getRootProps({ className: styles.zone })}>
      <input {...getInputProps()} />
      {isDragActive && (
        <div className={styles.dropIt}>
          <p>Drop it!</p>
        </div>
      )}
      <DropzoneContext.Provider value={{ openFilePicker }}>
        {children}
      </DropzoneContext.Provider>
    </div>
  );
}

export function useDropzoneContext() {
  return useContext(DropzoneContext);
}

export default Dropzone;
