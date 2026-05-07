import { render, screen } from '@testing-library/react';
import { FiGlobe } from 'react-icons/fi';
import { describe, expect, it } from 'vitest';

import Service from './Service';

describe('Service', () => {
  it('renders the heading when provided', () => {
    render(
      <Service icon={FiGlobe} heading="From URL">
        <span>body</span>
      </Service>,
    );
    expect(
      screen.getByRole('heading', { level: 2, name: 'From URL' }),
    ).toBeInTheDocument();
    expect(screen.getByText('body')).toBeInTheDocument();
  });

  it('omits the heading when not provided', () => {
    render(
      <Service icon={FiGlobe}>
        <span>body</span>
      </Service>,
    );
    expect(screen.queryByRole('heading')).not.toBeInTheDocument();
  });
});
