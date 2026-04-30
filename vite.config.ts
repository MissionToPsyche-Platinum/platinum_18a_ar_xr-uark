/** Vite config: React, wasm-pack for Rust/WASM, base path for deployment. */
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import wasmPack from 'vite-plugin-wasm-pack'

export default defineConfig({
  plugins: [
    react(),
    wasmPack('./rust_engine')
  ],
  // Update this to match your GitHub Pages repo slug (e.g. '/nasa-psyche-ar/'), or '/' for a custom domain.
  base: '/platinum_18a_ar_xr-uark/',
  server: {
    host: true,
  },
})
