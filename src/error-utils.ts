export function formatUnknownError(
  error: unknown,
  fallback = 'Unknown error',
): string {
  if (error instanceof Error) {
    return error.message;
  }

  if (typeof error === 'string') {
    return error;
  }

  if (
    typeof error === 'number' ||
    typeof error === 'boolean' ||
    typeof error === 'bigint'
  ) {
    return `${error}`;
  }

  if (typeof error === 'symbol') {
    return error.description ?? fallback;
  }

  try {
    const serialized: unknown = JSON.stringify(error);
    return typeof serialized === 'string' ? serialized : fallback;
  } catch {
    return fallback;
  }
}
