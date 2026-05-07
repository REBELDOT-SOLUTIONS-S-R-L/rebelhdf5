import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  plugins: [react()],
  define: {
    __PYTHON_BACKEND_PORT__: JSON.stringify(4095),
  },
  test: {
    globals: false,
    environment: 'jsdom',
    environmentOptions: {
      jsdom: {
        localStorage: true,
      },
    },
    setupFiles: ['./vitest.setup.ts'],
    include: ['{src,electron}/**/*.test.{ts,tsx,mjs}'],
    css: false,
    clearMocks: true,
    restoreMocks: true,
  },
});
