import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { useStore } from '../stores';
import SidebarToggle from './SidebarToggle';

const initialState = useStore.getState();

beforeEach(() => {
  globalThis.localStorage.clear();
});

afterEach(() => {
  useStore.setState(initialState, true);
});

describe('SidebarToggle', () => {
  it('shows "Collapse" label when expanded', () => {
    render(<SidebarToggle isCollapsed={false} />);
    expect(
      screen.getByRole('button', { name: 'Collapse sidebar' }),
    ).toBeInTheDocument();
  });

  it('shows "Expand" label when collapsed', () => {
    render(<SidebarToggle isCollapsed />);
    expect(
      screen.getByRole('button', { name: 'Expand sidebar' }),
    ).toBeInTheDocument();
  });

  it('toggles the store flag on click', async () => {
    render(<SidebarToggle isCollapsed={false} />);
    expect(useStore.getState().sidebarMayCollapse).toBe(false);

    await userEvent.click(screen.getByRole('button'));
    expect(useStore.getState().sidebarMayCollapse).toBe(true);

    await userEvent.click(screen.getByRole('button'));
    expect(useStore.getState().sidebarMayCollapse).toBe(false);
  });
});
