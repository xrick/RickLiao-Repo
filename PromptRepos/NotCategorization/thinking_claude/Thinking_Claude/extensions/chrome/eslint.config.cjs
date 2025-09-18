/** @type {import('eslint').Config[]} */
module.exports = [
  {
    ignores: [
      "**/dist/**",
      "**/node_modules/**",
      "**/coverage/**",
      "**/*.config.js",
    ],
  },
  require("@eslint/js").configs.recommended,
  {
    files: ["**/*.{ts,tsx,js,jsx}"],
    languageOptions: {
      ecmaVersion: 2021,
      sourceType: "module",
      parser: require("@typescript-eslint/parser"),
      parserOptions: {
        ecmaFeatures: {
          jsx: true,
        },
      },
      globals: {
        chrome: "readonly",
        console: "readonly",
        MutationObserver: "readonly",
        window: "readonly",
        document: "readonly",
        HTMLElement: "readonly",
        Element: "readonly",
        setTimeout: "readonly",
        navigator: "readonly",
        setInterval: "readonly",
        Node: "readonly",
        HTMLButtonElement: "readonly",
        MutationRecord: "readonly",
        MouseEvent: "readonly",
        SVGSVGElement: "readonly",
      },
    },
    plugins: {
      "@typescript-eslint": require("@typescript-eslint/eslint-plugin"),
      react: require("eslint-plugin-react"),
      "react-hooks": require("eslint-plugin-react-hooks"),
    },
    rules: {
      ...require("@typescript-eslint/eslint-plugin").configs.recommended.rules,
      ...require("eslint-plugin-react").configs.recommended.rules,
      ...require("eslint-plugin-react-hooks").configs.recommended.rules,
      "react/react-in-jsx-scope": "off",
      "@typescript-eslint/no-unused-vars": [
        "warn",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
        },
      ],
      "@typescript-eslint/explicit-function-return-type": "off",
      "@typescript-eslint/explicit-module-boundary-types": "off",
      "no-console": "off",
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "warn",
      "no-inline-styles": "off",
      "no-undef": "off",
      "react/prop-types": "off",
    },
    settings: {
      react: {
        version: "detect",
      },
    },
  },
  // Config for test files
  {
    files: ["**/__tests__/**/*", "**/*.test.*"],
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
      "no-console": "off",
    },
  },
  // Config for configuration files
  {
    files: ["*.config.js", "*.config.cjs", "webpack/**/*.js"],
    languageOptions: {
      globals: {
        module: "readonly",
        require: "readonly",
        __dirname: "readonly",
      },
    },
    rules: {
      "@typescript-eslint/no-require-imports": "off",
      "no-undef": "off",
    },
  },
  require("eslint-config-prettier"),
]
