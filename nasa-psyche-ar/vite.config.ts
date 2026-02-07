import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import wasmPack from 'vite-plugin-wasm-pack'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    wasmPack('./rust_engine')
  ],
  base: '/platinum_18a_ar_xr-uark/',
})
