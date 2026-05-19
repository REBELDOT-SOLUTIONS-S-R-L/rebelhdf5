import { describe, expect, it } from 'vitest';

import { FileService, type LocalFile, type RemoteFile } from './stores';
import {
  buildMailto,
  getDatasetAttributesLink,
  getDatasetProcessingLink,
  getFileLink,
  getObjectDistributionLink,
  getPoseTraceLink,
  getVideoConverterLink,
  getViewerLink,
  resolveFileUrl,
} from './utils';

describe('getFileLink helpers', () => {
  it('encodes the URL into the `url` query param for each viewer', () => {
    const href = 'https://example.com/file.h5';
    expect(getViewerLink(href)).toBe(
      '/view?url=https%3A%2F%2Fexample.com%2Ffile.h5',
    );
    expect(getPoseTraceLink(href)).toBe(
      '/pose-trace?url=https%3A%2F%2Fexample.com%2Ffile.h5',
    );
    expect(getVideoConverterLink(href)).toBe(
      '/video-converter?url=https%3A%2F%2Fexample.com%2Ffile.h5',
    );
    expect(getDatasetProcessingLink(href)).toBe(
      '/dataset-processing?url=https%3A%2F%2Fexample.com%2Ffile.h5',
    );
    expect(getDatasetAttributesLink(href)).toBe(
      '/dataset-attributes?url=https%3A%2F%2Fexample.com%2Ffile.h5',
    );
    expect(getObjectDistributionLink(href)).toBe(
      '/object-distribution?url=https%3A%2F%2Fexample.com%2Ffile.h5',
    );
  });

  it('escapes URL query characters', () => {
    expect(getFileLink('/view', 'https://x.com/a b?c=1&d=2')).toContain(
      'url=https%3A%2F%2Fx.com%2Fa+b%3Fc%3D1%26d%3D2',
    );
  });
});

describe('resolveFileUrl', () => {
  it('returns undefined for invalid URLs', async () => {
    await expect(resolveFileUrl('not-a-url')).resolves.toBeUndefined();
    await expect(resolveFileUrl('')).resolves.toBeUndefined();
  });

  it('returns undefined for non-https protocols', async () => {
    await expect(
      resolveFileUrl('http://example.com/x.h5'),
    ).resolves.toBeUndefined();
    await expect(
      resolveFileUrl('ftp://example.com/x.h5'),
    ).resolves.toBeUndefined();
    await expect(resolveFileUrl('blob:http://x/abc')).resolves.toBeUndefined();
  });

  it('detects raw GitHub URLs', async () => {
    const result = await resolveFileUrl(
      'https://raw.githubusercontent.com/user/repo/main/data.h5',
    );
    expect(result).toEqual({
      url: 'https://raw.githubusercontent.com/user/repo/main/data.h5',
      name: 'data.h5',
      service: FileService.GitHub,
      resolvedUrl: 'https://raw.githubusercontent.com/user/repo/main/data.h5',
    });
  });

  it('rewrites github.com /blob/ URLs to raw.githubusercontent.com', async () => {
    const result = await resolveFileUrl(
      'https://github.com/user/repo/blob/main/data.h5',
    );
    expect(result?.service).toBe(FileService.GitHub);
    expect(result?.resolvedUrl).toBe(
      'https://raw.githubusercontent.com/user/repo/main/data.h5',
    );
    expect(result?.name).toBe('data.h5');
  });

  it('detects GitLab subdomains', async () => {
    const result = await resolveFileUrl(
      'https://gitlab.example.com/user/repo/-/blob/main/data.h5',
    );
    expect(result?.service).toBe(FileService.GitLab);
    expect(result?.resolvedUrl).toBe(
      'https://gitlab.example.com/user/repo/-/raw/main/data.h5?inline=false',
    );
  });

  it('passes through generic URLs as the Url service', async () => {
    const result = await resolveFileUrl('https://example.com/data.h5');
    expect(result?.service).toBe(FileService.Url);
    expect(result?.resolvedUrl).toBe('https://example.com/data.h5');
  });

  it('uses the hostname when the pathname is empty', async () => {
    const result = await resolveFileUrl('https://example.com');
    expect(result?.name).toBe('example.com');
  });
});

describe('buildMailto', () => {
  it('produces a mailto link with the encoded subject and body', () => {
    const link = buildMailto('Issue', 'Something broke');
    expect(link.startsWith('mailto:h5web@esrf.fr?')).toBe(true);
    expect(link).toContain('subject=%5BmyHDF5%5D%20Issue');
    expect(link).toContain('Something%20broke');
  });

  it('uses %20 for spaces (not + signs)', () => {
    const link = buildMailto('Two words', 'word one');
    expect(link).not.toContain('+');
    expect(link).toContain('Two%20words');
  });

  it('includes the file URL when given a string', () => {
    const link = buildMailto('s', 'm', 'https://example.com/x.h5');
    expect(decodeURIComponent(link)).toContain(
      'File URL: https://example.com/x.h5',
    );
  });

  it('includes file metadata when given a remote H5File', () => {
    const file: RemoteFile = {
      url: 'https://example.com/x.h5',
      name: 'x.h5',
      service: FileService.GitHub,
      resolvedUrl: 'https://raw.githubusercontent.com/u/r/main/x.h5',
    };
    const decoded = decodeURIComponent(buildMailto('s', 'm', file));
    expect(decoded).toContain('File name: x.h5');
    expect(decoded).toContain('Service detected: GitHub');
    expect(decoded).toContain(
      'Resolved URL: https://raw.githubusercontent.com/u/r/main/x.h5',
    );
  });

  it('asks for the file when the file is local', () => {
    const file: LocalFile = {
      url: 'blob:abc',
      name: 'x.h5',
      service: FileService.Local,
      resolvedUrl: 'blob:abc',
      file: new File([], 'x.h5'),
    };
    const decoded = decodeURIComponent(buildMailto('s', 'm', file));
    expect(decoded).toContain('please send us your HDF5 file');
  });

  it('appends the entity path when provided', () => {
    const link = buildMailto('s', 'm', undefined, '/group/dataset');
    expect(decodeURIComponent(link)).toContain('Entity path: /group/dataset');
  });
});
