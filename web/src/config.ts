/**
 * Where the API lives.
 *
 * Default is same-origin, which is what happens when FastAPI serves the built
 * bundle (the `./run.sh` and Docker paths). Set `VITE_API_BASE` at build time
 * to point a separately-hosted front end — a Vercel deployment, say — at an
 * API running elsewhere.
 *
 *   VITE_API_BASE=https://api.example.com npm run build
 *
 * Note that Vercel can host this front end but NOT the Python API: an analysis
 * runs for 1-3 minutes and streams SSE throughout, which exceeds serverless
 * function limits. The API belongs on a long-lived host (EC2, Fly, Render).
 */

const raw = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

/** Normalised base URL with no trailing slash; "" means same-origin. */
export const API_BASE = raw.replace(/\/+$/, "");

/** Build an absolute URL for an API path. */
export function apiUrl(path: string): string {
  return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
}

/**
 * Plot URLs arrive from the server as root-relative paths (`/artifacts/...`).
 * When the API is on another origin they must be rewritten, or the browser
 * would request the image from the front end's own host and 404.
 */
export function assetUrl(path: string): string {
  if (!path) return path;
  if (/^https?:\/\//i.test(path)) return path;
  return apiUrl(path);
}
