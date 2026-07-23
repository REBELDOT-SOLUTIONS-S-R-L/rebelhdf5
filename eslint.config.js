import { createConfig, detectOpts } from '@esrf/eslint-config';
import { defineConfig, globalIgnores } from 'eslint/config';

const opts = detectOpts(import.meta.dirname);

const config = defineConfig([
  globalIgnores(['coverage/', 'dist/', 'electron/', '.venv/']),
  ...createConfig(opts),
]);

export default config;
