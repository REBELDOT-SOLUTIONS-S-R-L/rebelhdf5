import {
  FiActivity,
  FiAperture,
  FiCloud,
  FiColumns,
  FiDatabase,
  FiEye,
  FiFileText,
  FiFilm,
  FiLayers,
  FiPlusCircle,
} from 'react-icons/fi';
import { NavLink, useSearchParams } from 'react-router-dom';

import {
  getDatasetFeatureAvailability,
  useDatasetFeatureAvailability,
} from '../feature-availability';
import { useStore } from '../stores';
import {
  getDatasetAttributesLink,
  getDatasetComparisonLink,
  getDatasetProcessingLink,
  getObjectDistributionLink,
  getPoseTraceLink,
  getVideoConverterLink,
  getViewerLink,
} from '../utils';
import Flyout from './Flyout';
import OpenedFiles from './OpenedFiles';
import styles from './Sidebar.module.css';
import SidebarToggle from './SidebarToggle';

function Sidebar() {
  const opened = useStore((state) => state.opened);
  const mayCollapse = useStore((state) => state.sidebarMayCollapse);
  const isCollapsed = mayCollapse;
  const [searchParams] = useSearchParams();
  const activeFileUrl = searchParams.get('url') || opened.at(0)?.url;
  const activeFile =
    opened.find((openedFile) => openedFile.url === activeFileUrl) ?? null;
  const availability = useDatasetFeatureAvailability(opened);
  const poseTraceAvailability = getDatasetFeatureAvailability({
    file: activeFile,
    feature: 'poseTrace',
    opened,
    availability,
  });
  const videoConverterAvailability = getDatasetFeatureAvailability({
    file: activeFile,
    feature: 'videoConverter',
    opened,
    availability,
  });
  const datasetProcessingAvailability = getDatasetFeatureAvailability({
    file: activeFile,
    feature: 'datasetProcessing',
    opened,
    availability,
  });
  const datasetComparisonAvailability = getDatasetFeatureAvailability({
    file: activeFile,
    feature: 'datasetComparison',
    opened,
    availability,
  });
  const datasetAttributesAvailability = getDatasetFeatureAvailability({
    file: activeFile,
    feature: 'datasetAttributes',
    opened,
    availability,
  });
  const objectDistributionAvailability = getDatasetFeatureAvailability({
    file: activeFile,
    feature: 'objectDistribution',
    opened,
    availability,
  });

  return (
    <div className={styles.sidebar} data-collapsed={isCollapsed || undefined}>
      <div className={styles.sidebarInner}>
        <h1 className={styles.logo} data-reveal>
          <NavLink className={styles.logoLink} to="/">
            rebelHDF<span className={styles.logo5}>5</span>
          </NavLink>
        </h1>
        <nav className={styles.nav} data-reveal>
          <div className={styles.navItems}>
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

            {activeFile && poseTraceAvailability.status === 'available' ? (
              <NavLink
                className={styles.mainNavItem}
                to={getPoseTraceLink(activeFile.url)}
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
                title={poseTraceAvailability.reason}
                disabled
              >
                <FiActivity className={styles.icon} />
                <span className={styles.label}>Pose Trace</span>
              </button>
            )}

            {activeFile && videoConverterAvailability.status === 'available' ? (
              <NavLink
                className={styles.mainNavItem}
                to={getVideoConverterLink(activeFile.url)}
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
                title={videoConverterAvailability.reason}
                disabled
              >
                <FiFilm className={styles.icon} />
                <span className={styles.label}>Video Converter</span>
              </button>
            )}

            {activeFile &&
            datasetProcessingAvailability.status === 'available' ? (
              <NavLink
                className={styles.mainNavItem}
                to={getDatasetProcessingLink(activeFile.url)}
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
                title={datasetProcessingAvailability.reason}
                disabled
              >
                <FiLayers className={styles.icon} />
                <span className={styles.label}>Dataset Processing</span>
              </button>
            )}

            {activeFile &&
            datasetComparisonAvailability.status === 'available' ? (
              <NavLink
                className={styles.mainNavItem}
                to={getDatasetComparisonLink(activeFile.url)}
                aria-label="Dataset Comparison"
                title="Dataset Comparison"
              >
                <FiColumns className={styles.icon} />
                <span className={styles.label}>Dataset Comparison</span>
              </NavLink>
            ) : (
              <button
                type="button"
                className={styles.navBtn}
                aria-label="Dataset Comparison"
                title={datasetComparisonAvailability.reason}
                disabled
              >
                <FiColumns className={styles.icon} />
                <span className={styles.label}>Dataset Comparison</span>
              </button>
            )}

            {activeFile &&
            datasetAttributesAvailability.status === 'available' ? (
              <NavLink
                className={styles.mainNavItem}
                to={getDatasetAttributesLink(activeFile.url)}
                aria-label="Dataset Attributes"
                title="Dataset Attributes"
              >
                <FiDatabase className={styles.icon} />
                <span className={styles.label}>Dataset Attributes</span>
              </NavLink>
            ) : (
              <button
                type="button"
                className={styles.navBtn}
                aria-label="Dataset Attributes"
                title={datasetAttributesAvailability.reason}
                disabled
              >
                <FiDatabase className={styles.icon} />
                <span className={styles.label}>Dataset Attributes</span>
              </button>
            )}

            {activeFile &&
            objectDistributionAvailability.status === 'available' ? (
              <NavLink
                className={styles.mainNavItem}
                to={getObjectDistributionLink(activeFile.url)}
                aria-label="Object Distribution"
                title="Object Distribution"
              >
                <FiAperture className={styles.icon} />
                <span className={styles.label}>Object Distribution</span>
              </NavLink>
            ) : (
              <button
                type="button"
                className={styles.navBtn}
                aria-label="Object Distribution"
                title={objectDistributionAvailability.reason}
                disabled
              >
                <FiAperture className={styles.icon} />
                <span className={styles.label}>Object Distribution</span>
              </button>
            )}

            <NavLink
              className={styles.mainNavItem}
              to="/databricks"
              aria-label="Databricks"
              title="Databricks"
            >
              <FiCloud className={styles.icon} />
              <span className={styles.label}>Databricks</span>
            </NavLink>

            {isCollapsed && (
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
                  <OpenedFiles availability={availability} />
                </Flyout>
              </div>
            )}
          </div>

          {!isCollapsed && <OpenedFiles availability={availability} />}
        </nav>

        <div className={styles.footer}>
          <SidebarToggle isCollapsed={isCollapsed} />
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
