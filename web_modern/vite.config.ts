/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  base: "./", // 상대 경로 — FastAPI가 /detection/ 하위로 서빙해도 에셋(./assets) 정확 해석
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
      "/ws": { target: "ws://localhost:8000", ws: true },
      "/detection": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
  build: { outDir: "dist" },
  test: { environment: "jsdom", globals: true, setupFiles: ["./src/test-setup.ts"] },
});
