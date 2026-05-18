import { type ComponentType } from 'react';
import {
  FiDownload,
  FiGithub,
  FiGitlab,
  FiGlobe,
  FiMonitor,
  FiTrash2,
} from 'react-icons/fi';
import { TbLetterZ } from 'react-icons/tb';
import {
  createSearchParams,
  Link,
  useLocation,
  useNavigate,
  useSearchParams,
} from 'react-router-dom';
import { clear } from 'suspend-react';

import { FileService, type H5File, useStore } from '../stores';
import { getFileLink } from '../utils';
import sidebarStyles from './Sidebar.module.css';

const ICONS: Record<FileService, ComponentType<{ className?: string }>> = {
  [FileService.Local]: FiMonitor,
  [FileService.Url]: FiGlobe,
  [FileService.GitHub]: FiGithub,
  [FileService.GitLab]: FiGitlab,
  [FileService.Zenodo]: TbLetterZ,
};

function OpenedFiles() {
  const opened = useStore((state) => state.opened);
  const removeFileAt = useStore((state) => state.removeFileAt);

  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const fileUrl = searchParams.get('url');
  const fileRoute = (
    location.pathname === '/pose-trace'
      ? '/pose-trace'
      : location.pathname === '/video-converter'
        ? '/video-converter'
        : location.pathname === '/dataset-processing'
          ? '/dataset-processing'
          : location.pathname === '/dataset-attributes'
            ? '/dataset-attributes'
            : location.pathname === '/cloth-distribution'
              ? '/cloth-distribution'
              : '/view'
  );

  function removeFile(file: H5File, index: number, isActive: boolean) {
    if (isActive) {
      // Select next or previous file, or navigate back to homepage
      const nextIndex = index < opened.length - 1 ? index + 1 : index - 1;
      navigate(nextIndex >= 0 ? getFileLink(fileRoute, opened[nextIndex].url) : '/');
    }

    // Remove from store and evict from suspense cache
    removeFileAt(index);
    clear([file.url]);
  }

  return (
    <section className={sidebarStyles.openedFiles}>
      <h2 className={sidebarStyles.heading}>Opened files</h2>
      {opened.length > 0 ? (
        <ul className={`${sidebarStyles.navList} ${sidebarStyles.openedFilesList}`}>
          {opened.map((file, index) => {
            const { url, name, service, resolvedUrl } = file;
            const isActive = url === fileUrl;
            const Icon = ICONS[service];

            return (
              <li key={url} className={sidebarStyles.navListItem}>
                <Link
                  key={url}
                  className={sidebarStyles.navItem}
                  to={`${fileRoute}?${createSearchParams({ url }).toString()}`}
                  title={url}
                  aria-current={isActive ? 'page' : undefined}
                  onClick={(evt) => {
                    // Remove focus so flyout can hide itself
                    evt.currentTarget.blur();
                  }}
                >
                  <Icon className={sidebarStyles.icon} />
                  <span className={sidebarStyles.label}>{name}</span>
                </Link>
                <div className={sidebarStyles.actionBtnGroup}>
                  <a
                    className={sidebarStyles.downloadBtn}
                    href={resolvedUrl}
                    title={resolvedUrl}
                    download={name}
                    aria-label="Download file"
                    target="_blank"
                    rel="noreferrer"
                    onClick={(evt) => {
                      evt.stopPropagation();
                      evt.currentTarget.blur();
                    }}
                  >
                    <FiDownload />
                  </a>
                  <button
                    className={sidebarStyles.removeBtn}
                    type="button"
                    aria-label="Remove file"
                    onClick={(evt) => {
                      evt.preventDefault();
                      removeFile(file, index, isActive);
                    }}
                  >
                    <FiTrash2 />
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      ) : (
        <p className={sidebarStyles.hint}>To get started, please open a file</p>
      )}
    </section>
  );
}

export default OpenedFiles;
