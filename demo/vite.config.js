import { defineConfig } from 'vite-plus'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  base: process.env.GITHUB_ACTIONS ? '/capsolve-sp/' : '/',
  plugins: [react(), tailwindcss()],
  fmt: {
    semi: false,
    singleQuote: true,
  },
})
