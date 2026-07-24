import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import DatabricksPage from './DatabricksPage';
import {
  type pollBackendStatus,
  type PythonBackendStatus,
} from './python-backend';
import { useStore } from './stores';

const mocks = vi.hoisted(() => ({
  pollBackendStatus: vi.fn<typeof pollBackendStatus>(),
}));

vi.mock(import('./python-backend'), () => ({
  pollBackendStatus: mocks.pollBackendStatus,
  PYTHON_BACKEND_BASE_URL: 'http://127.0.0.1:4095',
}));

type FetchMock = ReturnType<typeof vi.fn<typeof fetch>>;
let fetchMock: FetchMock;

function requestUrl(input: RequestInfo | URL): string {
  if (typeof input === 'string') {
    return input;
  }
  return input instanceof URL ? input.href : input.url;
}

function requirePutSecretsCall(): Parameters<typeof fetch> {
  const call = fetchMock.mock.calls.find(([input]) =>
    requestUrl(input).includes('put-secrets'),
  );
  if (!call) {
    throw new Error('Expected a request to the put-secrets endpoint.');
  }
  return call;
}

function requireStringBody(init: RequestInit | undefined): string {
  if (typeof init?.body !== 'string') {
    throw new TypeError('Expected the request body to be a string.');
  }
  return init.body;
}

function renderPage(): void {
  render(
    <MemoryRouter initialEntries={['/databricks']}>
      <DatabricksPage />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  useStore.setState({ opened: [] }, false);
  mocks.pollBackendStatus.mockImplementation(
    (onStatus: (status: PythonBackendStatus) => void) => {
      onStatus({ available: true, rootDir: '/data', version: 3 });
      return () => {};
    },
  );

  fetchMock = vi.fn<typeof fetch>(async (input: RequestInfo | URL) => {
    const url = requestUrl(input);
    if (url.includes('active-runs')) {
      return Response.json({ runs: [] });
    }
    if (url.includes('put-secrets')) {
      return Response.json({ allOk: true, results: [] });
    }
    if (url.includes('volume-files')) {
      return Response.json({ files: [] });
    }
    return Response.json({});
  });
  vi.stubGlobal('fetch', fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
  useStore.setState({ opened: [] }, false);
});

describe('DatabricksPage', () => {
  it('renders the Secrets section', () => {
    renderPage();
    expect(
      screen.getByRole('heading', { name: /Secrets/u }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /Push Secrets/u }),
    ).toBeInTheDocument();
  });

  it('POSTs entered secrets to the brev scope', async () => {
    const user = userEvent.setup();
    renderPage();

    await user.type(screen.getByPlaceholderText('e.g. 5000'), '5000');
    await user.type(screen.getByPlaceholderText('Brev API token'), 'tok-123');
    await user.click(screen.getByRole('button', { name: /Push Secrets/u }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        'http://127.0.0.1:4095/api/databricks/put-secrets',
        expect.objectContaining({ method: 'POST' }),
      );
    });

    const [, init] = requirePutSecretsCall();
    const body: unknown = JSON.parse(requireStringBody(init));
    expect(body).toEqual({
      secrets: { max_steps: '5000', token: 'tok-123' },
      scope: 'brev',
    });

    await expect(
      screen.findByText(/Saved 2 secret/u),
    ).resolves.toBeInTheDocument();
  });
});
