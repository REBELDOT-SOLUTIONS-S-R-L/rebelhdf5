import { useEffect, useMemo, useState } from 'react';

import { selectDatasetGroup } from './ObjectDistributionPage.utils';
import { inspectSourceFeatures, openPoseTraceSource } from './pose-trace/hdf5';
import { pollBackendStatus } from './python-backend';
import { FileService, type H5File } from './stores';

export type DatasetFeature =
  | 'poseTrace'
  | 'videoConverter'
  | 'datasetProcessing'
  | 'datasetComparison'
  | 'datasetAttributes'
  | 'objectDistribution';

export type AvailabilityStatus = 'pending' | 'available' | 'unavailable';

export interface FeatureAvailability {
  status: AvailabilityStatus;
  reason: string;
}

export type FileFeatureAvailability = Record<
  Exclude<DatasetFeature, 'datasetAttributes'>,
  FeatureAvailability
>;

export interface AvailabilityState {
  byUrl: Partial<Record<string, FileFeatureAvailability>>;
  backendAvailable: boolean | null;
}

const PENDING_REASON = 'Inspecting dataset schema...';

const pending: FeatureAvailability = {
  status: 'pending',
  reason: PENDING_REASON,
};

const available: FeatureAvailability = {
  status: 'available',
  reason: '',
};

function noop() {
  return undefined;
}

function unavailable(reason: string): FeatureAvailability {
  return { status: 'unavailable', reason };
}

function formatUnknownError(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }

  if (typeof error === 'string') {
    return error;
  }

  return 'Unknown inspection error';
}

function buildPendingAvailability(): FileFeatureAvailability {
  return {
    poseTrace: pending,
    videoConverter: pending,
    datasetProcessing: pending,
    datasetComparison: pending,
    objectDistribution: pending,
  };
}

function buildUnavailableAvailability(reason: string): FileFeatureAvailability {
  const result = unavailable(reason);
  return {
    poseTrace: result,
    videoConverter: result,
    datasetProcessing: result,
    datasetComparison: result,
    objectDistribution: result,
  };
}

async function inspectFile(file: H5File): Promise<FileFeatureAvailability> {
  let cleanup: () => void = noop;

  try {
    const source = await openPoseTraceSource(file);
    ({ cleanup } = source);
    const capabilities = await inspectSourceFeatures(source);
    const hasDemos = capabilities.demoCount > 0;
    const hasKeys = hasDemos && capabilities.keyCount > 0;

    return {
      poseTrace:
        capabilities.poseTraceDemoCount > 0
          ? available
          : unavailable('No usable pose, joint, or object trace data found.'),
      videoConverter:
        capabilities.videoDemoCount > 0
          ? available
          : unavailable('No supported video datasets found.'),
      datasetProcessing: hasKeys
        ? available
        : unavailable('No demo datasets found to process.'),
      datasetComparison: hasKeys
        ? available
        : unavailable('No demo datasets found to compare.'),
      objectDistribution:
        capabilities.objectDistributionDemoCount > 0
          ? available
          : unavailable('No usable object initial-pose data found.'),
    };
  } catch (error: unknown) {
    return buildUnavailableAvailability(formatUnknownError(error));
  } finally {
    cleanup();
  }
}

const inspectionCache = new Map<
  string,
  Promise<FileFeatureAvailability> | FileFeatureAvailability
>();

async function getCachedInspection(
  file: H5File,
): Promise<FileFeatureAvailability> {
  const cached = inspectionCache.get(file.url);
  if (cached) {
    return cached;
  }

  const promise = inspectFile(file);
  inspectionCache.set(file.url, promise);
  const availability = await promise;
  if (inspectionCache.get(file.url) === promise) {
    inspectionCache.set(file.url, availability);
  }
  return availability;
}

export function useDatasetFeatureAvailability(
  opened: H5File[],
): AvailabilityState {
  const [byUrl, setByUrl] = useState<
    Partial<Record<string, FileFeatureAvailability>>
  >({});
  const [backendAvailable, setBackendAvailable] = useState<boolean | null>(
    null,
  );

  useEffect(() => {
    return pollBackendStatus((status) => {
      setBackendAvailable(status.available);
    });
  }, []);

  useEffect(() => {
    let cancelled = false;
    const openedUrls = new Set(opened.map((file) => file.url));

    setByUrl((current) => {
      const next: Partial<Record<string, FileFeatureAvailability>> =
        Object.fromEntries(
          Object.entries(current).filter(([url]) => openedUrls.has(url)),
        );

      for (const file of opened) {
        if (!(file.url in next)) {
          const cached = inspectionCache.get(file.url);
          next[file.url] =
            cached && !(cached instanceof Promise)
              ? cached
              : buildPendingAvailability();
        }
      }

      return next;
    });

    async function inspectOpenedFiles() {
      await Promise.all(
        opened.map(async (file) => {
          const fileAvailability = await getCachedInspection(file);
          if (cancelled) {
            return;
          }

          setByUrl((current) => {
            if (!openedUrls.has(file.url)) {
              return current;
            }

            return { ...current, [file.url]: fileAvailability };
          });
        }),
      );
    }

    void inspectOpenedFiles();

    return () => {
      cancelled = true;
    };
  }, [opened]);

  return useMemo(
    () => ({ byUrl, backendAvailable }),
    [backendAvailable, byUrl],
  );
}

function getDatasetAttributesAvailability(
  file: H5File,
  availability: AvailabilityState,
): FeatureAvailability {
  if (file.service !== FileService.Local) {
    return unavailable('Dataset attributes require a local Electron file.');
  }

  if (!file.serverPath) {
    return unavailable('Reopen this file so its filesystem path is known.');
  }

  const processingAvailability =
    availability.byUrl[file.url]?.datasetProcessing ?? pending;
  if (processingAvailability.status !== 'available') {
    return processingAvailability.status === 'pending'
      ? processingAvailability
      : unavailable('Dataset attributes require a standard demo dataset.');
  }

  if (availability.backendAvailable === null) {
    return pending;
  }

  return availability.backendAvailable
    ? available
    : unavailable('Python backend is not available.');
}

export function getDatasetFeatureAvailability({
  file,
  feature,
  opened,
  availability,
}: {
  file: H5File | null;
  feature: DatasetFeature;
  opened: H5File[];
  availability: AvailabilityState;
}): FeatureAvailability {
  if (!file) {
    return unavailable('Open a dataset first.');
  }

  if (feature === 'datasetAttributes') {
    return getDatasetAttributesAvailability(file, availability);
  }

  if (feature === 'datasetComparison') {
    return available;
  }

  if (feature === 'objectDistribution') {
    const selection = selectDatasetGroup(opened, file.url);
    if (!selection.successUrl || !selection.failedUrl) {
      return unavailable(
        'Open matching successful and failed generated datasets.',
      );
    }

    for (const url of [selection.successUrl, selection.failedUrl]) {
      const status = availability.byUrl[url]?.objectDistribution ?? pending;
      if (status.status !== 'available') {
        return status;
      }
    }

    return available;
  }

  return availability.byUrl[file.url]?.[feature] ?? pending;
}
