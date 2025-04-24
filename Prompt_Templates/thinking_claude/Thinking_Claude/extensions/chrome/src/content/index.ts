import "@/styles/index.css"

import { ExtensionManager } from "./v3/managers/extension-manager"

// Create a single instance of ExtensionManager
const extensionManager = new ExtensionManager()
// Initialize on first load
extensionManager.initialize()

// Listen for route changes from background script
chrome.runtime.onMessage.addListener(
  (message: { type: string; url: string }) => {
    if (message.type === "ROUTE_CHANGED") {
      // Reinitialize the extension for the new route
      extensionManager.cleanup()
      extensionManager.initialize()
    }
  }
)
