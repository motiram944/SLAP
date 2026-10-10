/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        slap: {
          bg: "#070b14",
          surface: "#0d1424",
          card: "#121d33",
          border: "#1e2d4d",
          cyan: "#06b6d4",
          emerald: "#10b981",
          violet: "#8b5cf6",
          amber: "#f59e0b",
          rose: "#f43f5e",
        },
      },
      fontFamily: {
        mono: ["JetBrains Mono", "Menlo", "monospace"],
      },
      animation: {
        pulseGlow: "pulseGlow 2.5s infinite ease-in-out",
      },
      keyframes: {
        pulseGlow: {
          "0%, 100%": { opacity: 0.4 },
          "50%": { opacity: 0.8 },
        },
      },
    },
  },
  plugins: [],
};
