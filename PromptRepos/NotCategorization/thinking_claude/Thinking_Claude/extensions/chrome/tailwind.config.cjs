/**Know Issue for now:
 * Our own Tailwind CSS is not being bundled into the build.
 * TODO: Figure out why and fix this later.
 */
const colors = require("tailwindcss/colors")

/** @type {import('tailwindcss').Config} */
module.exports = {
  prefix: "tc-",
  important: true,
  content: [
    "./src/**/*.{ts,tsx}",
    "./src/components/**/*.{ts,tsx}",
    "./src/content/**/*.{ts,tsx}",
    "./src/**/*.{ts,tsx,html,js,jsx}",
    "./public/**/*.html",
  ],
  theme: {
    extend: {
      colors: {
        border: "hsl(var(--tc-border))",
        input: "hsl(var(--tc-input))",
        ring: "hsl(var(--tc-ring))",
        background: "hsl(var(--tc-background))",
        foreground: "hsl(var(--tc-foreground))",
        primary: {
          DEFAULT: "hsl(var(--tc-primary))",
          foreground: "hsl(var(--tc-primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--tc-secondary))",
          foreground: "hsl(var(--tc-secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--tc-destructive))",
          foreground: "hsl(var(--tc-destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--tc-muted))",
          foreground: "hsl(var(--tc-muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--tc-accent))",
          foreground: "hsl(var(--tc-accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--tc-popover))",
          foreground: "hsl(var(--tc-popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--tc-card))",
          foreground: "hsl(var(--tc-card-foreground))",
        },
        // Include Tailwind's default colors
        slate: colors.slate,
        gray: colors.gray,
        zinc: colors.zinc,
        neutral: colors.neutral,
        stone: colors.stone,
        red: colors.red,
        orange: colors.orange,
        amber: colors.amber,
        yellow: colors.yellow,
        lime: colors.lime,
        green: colors.green,
        emerald: colors.emerald,
        teal: colors.teal,
        cyan: colors.cyan,
        sky: colors.sky,
        blue: colors.blue,
        indigo: colors.indigo,
        violet: colors.violet,
        purple: colors.purple,
        fuchsia: colors.fuchsia,
        pink: colors.pink,
        rose: colors.rose,
      },
      //   we can inherit the original border radius form claude here
      //   borderRadius: {
      //     lg: "`var(--radius)`",
      //     md: "`calc(var(--radius) - 2px)`",
      //     sm: "calc(var(--radius) - 4px)",
      //   },
      keyframes: {
        "accordion-down": {
          from: {
            height: "0",
          },
          to: {
            height: "var(--radix-accordion-content-height)",
          },
        },
        "accordion-up": {
          from: {
            height: "var(--radix-accordion-content-height)",
          },
          to: {
            height: "0",
          },
        },
        shimmer: {
          "0%": { backgroundPosition: "200% 50%" },
          "100%": { backgroundPosition: "-200% 50%" },
        },
        "fade-in": {
          "0%": { opacity: "0", transform: "translateY(0)" },
          "20%": { opacity: "0.2", transform: "translateY(0)" },
          "50%": { opacity: "0.5", transform: "translateY(0)" },
          "60%": { opacity: "0.6", transform: "translateY(0)" },
          "80%": { opacity: "0.8", transform: "translateY(0)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
        shimmer: "shimmer 3s linear infinite",
        "fade-in": "fade-in 0.45s ease-out",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
}
