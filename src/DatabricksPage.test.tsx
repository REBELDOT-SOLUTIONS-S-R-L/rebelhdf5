import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import DatabricksPage from './DatabricksPage';
import { useStore } from './stores';

const mocks = vi.hoisted(() => ({
  pollBackendStatus: vi.fn(),
}));

vi.mock('./python-backend', () => ({
  pollBackendStatus: mocks.pollBackendStatus,
  PYTHON_BACKEND_BASE_URL: 'http://127.0.0.1:4095',
}));

type FetchMock = ReturnType<typeof vi.fn<typeof fetch>>;
let fetchMock: FetchMock;

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
    (onStatus: (s: { available: boolean }) => void) => {
      onStatus({ available: true });
      return () => {};
    },
  );

  fetchMock = vi.fn<typeof fetch>(async (input: RequestInfo | URL) => {
    const url = String(input);
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

    const call = fetchMock.mock.calls.find(([url]) =>
      String(url).includes('put-secrets'),
    );
    expect(call).toBeDefined();
    const body = JSON.parse((call![1] as RequestInit).body as string);
    expect(body).toEqual({
      secrets: { max_steps: '5000', token: 'tok-123' },
      scope: 'brev',
    });

    expect(await screen.findByText(/Saved 2 secret/u)).toBeInTheDocument();
  });
});
