import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://tauri.app/develop/#frontend
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 5173,
    strictPort: true,
  },
  envPrefix: ["VITE_", "TAURI_"],
  build: {
    target: process.env.TAURI_ENV_PLATFORM === "windows" ? "chrome105" : "safari13",
    outDir: "dist",
  },
});
