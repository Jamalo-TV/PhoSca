import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const certDir = resolve(import.meta.dirname, "..", ".cert");
const https = process.env.PHOSCA_HTTP_DEV !== "true" && existsSync(resolve(certDir, "phosca-key.pem"))
  ? { key: readFileSync(resolve(certDir, "phosca-key.pem")), cert: readFileSync(resolve(certDir, "phosca-cert.pem")) }
  : undefined;

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5173,
    https,
    proxy: {
      "/api": "http://127.0.0.1:8000",
      "/metrics": "http://127.0.0.1:8000",
    },
  },
  preview: {
    host: "0.0.0.0",
    port: 4173,
    https,
  },
});
