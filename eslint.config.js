import { createConfig, detectOpts } from '@esrf/eslint-config';
import { defineConfig, globalIgnores } from 'eslint/config';

const opts = detectOpts(import.meta.dirname);
// ESLint suppressions record error-level violations only. Promote the shared
// preset's warnings so the checked-in baseline can cover existing lint debt
// while still rejecting any new warning or error beyond the recorded counts.
const sharedConfig = createConfig(opts).map((entry) => {
  if (!entry.rules) {
    return entry;
  }

  return {
    ...entry,
    rules: Object.fromEntries(
      Object.entries(entry.rules).map(([ruleName, setting]) => {
        if (setting === 'warn' || setting === 1) {
          return [ruleName, 'error'];
        }
        if (
          Array.isArray(setting) &&
          (setting[0] === 'warn' || setting[0] === 1)
        ) {
          return [ruleName, ['error', ...setting.slice(1)]];
        }
        return [ruleName, setting];
      }),
    ),
  };
});

const config = defineConfig([
  globalIgnores(['coverage/', 'dist/', 'electron/', '.venv/']),
  ...sharedConfig,
]);

export default config;
