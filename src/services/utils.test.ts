import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  fetchZenodoFileUrl,
  toRawGithubUrl,
  toRawGitlabHref,
  validateRequiredUrl,
} from './utils';

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('toRawGithubUrl', () => {
  it('rewrites a github.com /blob/ URL to raw.githubusercontent.com', () => {
    expect(
      toRawGithubUrl('https://github.com/user/repo/blob/main/file.h5'),
    ).toBe('https://raw.githubusercontent.com/user/repo/main/file.h5');
  });

  it('strips the ?raw=true suffix', () => {
    expect(
      toRawGithubUrl('https://github.com/user/repo/blob/main/file.h5?raw=true'),
    ).toBe('https://raw.githubusercontent.com/user/repo/main/file.h5');
  });
});

describe('toRawGitlabHref', () => {
  it('replaces /blob/ with /raw/ and forces inline=false', () => {
    const url = new URL(
      'https://gitlab.example.com/user/repo/-/blob/main/data.h5',
    );
    expect(toRawGitlabHref(url)).toBe(
      'https://gitlab.example.com/user/repo/-/raw/main/data.h5?inline=false',
    );
  });

  it('does not mutate the input URL', () => {
    const url = new URL(
      'https://gitlab.example.com/user/repo/-/blob/main/data.h5',
    );
    const before = url.href;
    toRawGitlabHref(url);
    expect(url.href).toBe(before);
  });
});

describe('fetchZenodoFileUrl', () => {
  it('throws when the URL does not match the Zenodo record pattern', async () => {
    await expect(fetchZenodoFileUrl('https://zenodo.org/foo')).rejects.toThrow(
      /not recognised/u,
    );
  });

  it('queries the Zenodo API and returns the matched file self link', async () => {
    const fetchMock = vi.fn<typeof fetch>(async () =>
      Response.json(
        {
          files: [
            {
              key: 'other.h5',
              links: {
                self: 'https://zenodo.org/api/records/1/files/other.h5/content',
              },
            },
            {
              key: 'data.h5',
              links: {
                self: 'https://zenodo.org/api/records/1/files/data.h5/content',
              },
            },
          ],
        },
        { status: 200 },
      ),
    );
    vi.stubGlobal('fetch', fetchMock);

    const result = await fetchZenodoFileUrl(
      'https://zenodo.org/record/1/files/data.h5',
    );

    expect(result).toBe(
      'https://zenodo.org/api/records/1/files/data.h5/content',
    );
    expect(fetchMock).toHaveBeenCalledWith('https://zenodo.org/api/records/1');
  });

  it('throws when the Zenodo response does not contain the file', async () => {
    vi.stubGlobal('fetch', async () =>
      Response.json({ files: [] }, { status: 200 }),
    );
    await expect(
      fetchZenodoFileUrl('https://zenodo.org/records/1/files/data.h5'),
    ).rejects.toThrow(/File not found/u);
  });
});

describe('validateRequiredUrl', () => {
  it('rejects empty input', () => {
    expect(validateRequiredUrl('')).toBe('Please enter a URL');
  });

  it('rejects unparseable URLs', () => {
    expect(validateRequiredUrl('not a url')).toBe(
      'Please enter a valid URL starting with https://',
    );
  });

  it('rejects non-https URLs', () => {
    expect(validateRequiredUrl('http://example.com/x.h5')).toBe(
      'The URL must start with https://',
    );
  });

  it('accepts https URLs', () => {
    expect(validateRequiredUrl('https://example.com/x.h5')).toBe(true);
  });
});
