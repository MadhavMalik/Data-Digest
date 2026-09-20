import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The built bundle lands in ui/dist, which the FastAPI app serves. One process
// serves both the API and the site, so deployment is a single command.
const API = process.env.SIGNAL_API ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../ui/dist",
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    port: 5173,
    // `npm run dev` proxies to the API so local development needs no CORS.
    proxy: Object.fromEntries(
      ["/analyses", "/datasets", "/artifacts", "/evidence", "/metrics", "/health"].map(
        (route) => [route, { target: API, changeOrigin: true }],
      ),
    ),
  },
});
