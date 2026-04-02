import { FiActivity, FiEye, FiFileText, FiFilm, FiLayers, FiPlusCircle } from 'react-icons/fi';
import { NavLink, useMatch, useSearchParams } from 'react-router-dom';

import { useStore } from '../stores';
import {
  getDatasetProcessingLink,
  getPoseTraceLink,
  getVideoConverterLink,
  getViewerLink,
} from '../utils';
import Flyout from './Flyout';
import OpenedFiles from './OpenedFiles';
import styles from './Sidebar.module.css';
import SidebarToggle from './SidebarToggle';

function Sidebar() {
  const isViewingFile = !!useMatch('/view');
  const opened = useStore((state) => state.opened);
  const mayCollapse = useStore((state) => state.sidebarMayCollapse);
  const isCollapsed = mayCollapse && isViewingFile;
  const [searchParams] = useSearchParams();
  const activeFileUrl = searchParams.get('url') ?? opened[0]?.url ?? null;

  return (
    <div className={styles.sidebar} data-collapsed={isCollapsed || undefined}>
      <div className={styles.sidebarInner}>
        <h1 className={styles.logo} data-reveal>
          rebelHDF<span>5</span>
        </h1>
        <nav className={styles.nav} data-reveal>
          <NavLink
            className={styles.mainNavItem}
            to="/"
            aria-label="Open HDF5"
            title="Open HDF5"
            data-primary
          >
            <FiPlusCircle className={styles.icon} />
            <span className={styles.label}>Open HDF5</span>
          </NavLink>

          {activeFileUrl ? (
            <NavLink
              className={styles.mainNavItem}
              to={getViewerLink(activeFileUrl)}
              aria-label="Viewer"
              title="Viewer"
            >
              <FiEye className={styles.icon} />
              <span className={styles.label}>Viewer</span>
            </NavLink>
          ) : (
            <button
              type="button"
              className={styles.navBtn}
              aria-label="Viewer"
              title="Viewer"
              disabled
            >
              <FiEye className={styles.icon} />
              <span className={styles.label}>Viewer</span>
            </button>
          )}

          {activeFileUrl ? (
            <NavLink
              className={styles.mainNavItem}
              to={getPoseTraceLink(activeFileUrl)}
              aria-label="Pose Trace"
              title="Pose Trace"
            >
              <FiActivity className={styles.icon} />
              <span className={styles.label}>Pose Trace</span>
            </NavLink>
          ) : (
            <button
              type="button"
              className={styles.navBtn}
              aria-label="Pose Trace"
              title="Pose Trace"
              disabled
            >
              <FiActivity className={styles.icon} />
              <span className={styles.label}>Pose Trace</span>
            </button>
          )}

          {activeFileUrl ? (
            <NavLink
              className={styles.mainNavItem}
              to={getVideoConverterLink(activeFileUrl)}
              aria-label="Video Converter"
              title="Video Converter"
            >
              <FiFilm className={styles.icon} />
              <span className={styles.label}>Video Converter</span>
            </NavLink>
          ) : (
            <button
              type="button"
              className={styles.navBtn}
              aria-label="Video Converter"
              title="Video Converter"
              disabled
            >
              <FiFilm className={styles.icon} />
              <span className={styles.label}>Video Converter</span>
            </button>
          )}

          {activeFileUrl ? (
            <NavLink
              className={styles.mainNavItem}
              to={getDatasetProcessingLink(activeFileUrl)}
              aria-label="Dataset Processing"
              title="Dataset Processing"
            >
              <FiLayers className={styles.icon} />
              <span className={styles.label}>Dataset Processing</span>
            </NavLink>
          ) : (
            <button
              type="button"
              className={styles.navBtn}
              aria-label="Dataset Processing"
              title="Dataset Processing"
              disabled
            >
              <FiLayers className={styles.icon} />
              <span className={styles.label}>Dataset Processing</span>
            </button>
          )}

          {isCollapsed ? (
            <div className={styles.flyoutWrapper}>
              <button
                type="button"
                className={styles.flyoutBtn}
                aria-label="Opened files"
                aria-current="true"
              >
                <FiFileText />
              </button>
              <Flyout>
                <OpenedFiles />
              </Flyout>
            </div>
          ) : (
            <OpenedFiles />
          )}
        </nav>

        <div className={styles.footer}>
          <SidebarToggle
            isCollapsed={isCollapsed}
            isDisabled={!isViewingFile}
          />
          <p className={styles.credits} data-reveal>
            Made by{' '}
            <a href="https://www.panosc.eu/" target="_blank" rel="noreferrer">
              PaNOSC
            </a>{' '}
            at&nbsp;
            <a href="https://www.esrf.fr/" target="_blank" rel="noreferrer">
              ESRF
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}

export default Sidebar;
