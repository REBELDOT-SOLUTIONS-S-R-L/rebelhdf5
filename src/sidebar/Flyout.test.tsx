import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import Flyout from './Flyout';

describe('Flyout', () => {
  it('renders its children inside the flyout container', () => {
    render(
      <Flyout>
        <span data-testid="payload">contents</span>
      </Flyout>,
    );
    expect(screen.getByTestId('payload')).toBeInTheDocument();
  });
});
