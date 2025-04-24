# Thinking Claude Chrome Extension

A Chrome extension that enhances Claude's thinking process, making it more human-like and transparent.

> **Important Notice**: The original Chrome extension (`chrome_v0`) has been deprecated. This is the new rewritten version (`chrome`) with improved architecture and modern tech stack. If you're using the old version, please update to this new version for better performance and continued support.

## How to Use 🚀

### Option 1: Direct Installation (Recommended)

1. **Download the Extension**

   - Go to [Latest Releases](https://github.com/richards199999/Thinking-Claude/releases)
   - Download the latest version (e.g., `thinking-claude-v1.0.2.zip`)
   - Extract the ZIP file

2. **Install in Chrome**

   - Open Chrome and go to `chrome://extensions/`
   - Enable "Developer mode" in the top right
   - Click "Load unpacked"
   - Select the `dist` folder in the extracted folder

3. **Start Using**
   - Visit [Claude.ai](https://claude.ai)
   - Start a new conversation or refresh an existing one
   - The extension will automatically enhance Claude's thinking process

### Option 2: Build Locally (For Development)

1. **Quick Setup**

   ```bash
   # Clone the repository
   git clone https://github.com/richards199999/Thinking-Claude.git
   cd Thinking-Claude/extensions/chrome

   # Install dependencies
   bun install

   # Build the extension
   bun run build
   ```

2. **Load in Chrome**

   - Open Chrome and go to `chrome://extensions/`
   - Enable "Developer mode" in the top right
   - Click "Load unpacked"
   - Select the `dist` folder (created after building)

3. **Development Mode**

   ```bash
   # Start development server with hot reload
   bun run start

   # Watch for changes
   bun run watch
   ```

## Tech Stack 🛠️

### Core Technologies

- **Language & Type Safety**

  - [TypeScript](https://www.typescriptlang.org/) - Strongly typed programming language
  - [ESLint](https://eslint.org/) - Code linting and standards
  - [Prettier](https://prettier.io/) - Code formatting

- **Frontend**
  - [React](https://react.dev/) - UI library
  - [Tailwind CSS](https://tailwindcss.com/) - Utility-first CSS framework
  - [shadcn/ui](https://ui.shadcn.com/) - Best UI components
  - [Chrome Extension API](https://developer.chrome.com/docs/extensions/) - Browser extension development

### Development Tools

- **Build & Bundle**

  - [Bun](https://bun.sh) - JavaScript all-in-one toolkit
  - [Webpack](https://webpack.js.org/) - Module bundler
  - [PostCSS](https://postcss.org/) - CSS processing

- **Testing & Quality**

  - [Vitest](https://vitest.dev/) - Unit testing framework
  - [Husky](https://typicode.github.io/husky/) - Git hooks
  - [lint-staged](https://github.com/okonet/lint-staged) - Staged files linter

- **Development Environment**
  - [Node.js](https://nodejs.org/) - JavaScript runtime
  - [Chrome DevTools](https://developer.chrome.com/docs/devtools/) - Browser debugging

## Getting Started with development 🚀

### What You'll Need

Required tools:

- [Bun](https://bun.sh) - A fast all-in-one JavaScript runtime & toolkit
- [Node.js](https://nodejs.org/) (v18 or higher) - JavaScript runtime environment
- [Git](https://git-scm.com/downloads) - For version control
- [Google Chrome](https://www.google.com/chrome/) - The browser we're building for

This extension uses:

- [TypeScript](https://www.typescriptlang.org/) - Type-safe JavaScript
- [React](https://react.dev/) - UI framework
- [Tailwind CSS](https://tailwindcss.com/) - Utility-first CSS framework
- [Webpack](https://webpack.js.org/) - Module bundler

#### Installing Node.js

1. Download Node.js from [nodejs.org](https://nodejs.org/)
2. Choose the LTS (Long Term Support) version
3. Run the installer
4. Verify installation:

   ```bash
   node --version
   npm --version
   ```

#### Installing Bun

Bun is required to run this project. Here's how to install it:

**Windows Users:**

1. First, install Windows Subsystem for Linux (WSL):

   ```powershell
   # Open PowerShell as Administrator and run:
   wsl --install
   ```

   After installation, restart your computer.

2. Install Bun through WSL:

   ```bash
   # Open WSL terminal and run:
   curl -fsSL https://bun.sh/install | bash
   ```

**macOS or Linux Users:**

```bash
# Open terminal and run:
curl -fsSL https://bun.sh/install | bash
```

To verify installation, run:

```bash
bun --version
```

### Setting Up Your Development Environment

1. Get the code:

   ```bash
   # Clone this repository to your computer
   git clone https://github.com/richards199999/Thinking-Claude.git

   # Go to the extension directory
   cd extensions/chrome

   # Install project dependencies
   bun install
   ```

### Development Commands

Here are the main commands you'll use during development:

```bash
# Build the extension for production
bun run build

# Start development mode with auto-reload
bun run start

# Watch for file changes
bun run watch

# Run tests
bun run test

# Fix code style and formatting
bun run fix
```

### Installing the Extension in Chrome

1. Open Chrome and type `chrome://extensions/` in the address bar
2. Turn on "Developer mode" using the switch in the top right corner
3. Click "Load unpacked" and select the `dist (visible after running bun run build)` folder from this project

## Project Organization 📁

```
chrome/
├── src/             # Your source code goes here
├── public/            # Built extension (created after running build)
│   ├── manifest.json # Extension configuration
│   ├── content.js   # Main extension script
│   └── icons/       # Extension icons
├── package.json     # Project configuration and scripts
└── CHANGELOG.md     # Version history and changes
```

## Development Workflow 🔄

### Code Quality Tools

We use several tools to maintain code quality:

- **Husky**: Automatically checks your code before commits
- **ESLint**: Finds and fixes JavaScript problems
- **Prettier**: Formats your code consistently

### Version Control & Releases

The project uses automated version bumping through CI:

- **Automatic Version Bumping**: When code is merged to main, the CI will:

  - Auto-increment the patch version (e.g., 1.0.0 -> 1.0.1)
  - Create a new release with the bumped version
  - Skip version bump for major versions (x.0.0)

- **Manual Version Control**:
  - Developers can manually set both versions in `package.json` and `manifest.json`
  - Major version changes (x.0.0) must be set manually
  - Manual versions will be respected by CI

> **Note**: If you need to manually set a version, update both `package.json` and `manifest.json` before merging to main.

### Continuous Integration

Our GitHub Actions setup automatically:

- Builds the extension
- Updates version numbers
- Creates new releases

## Need Help? 🤔

- Check the [CHANGELOG.md](./CHANGELOG.md) for recent updates
- Visit our [GitHub Issues](https://github.com/richards199999/Thinking-Claude/issues) for known problems or to report new ones
- Feel free to ask questions in our GitHub Discussions
