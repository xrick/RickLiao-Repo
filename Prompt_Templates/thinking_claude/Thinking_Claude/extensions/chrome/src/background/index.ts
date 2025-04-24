// Message types for type safety
type RouteChangeMessage = {
  type: "ROUTE_CHANGED"
  url: string
}

// Track the last URL to prevent duplicate notifications
let lastUrl: string | null = null

// Function to notify content script about route changes
function notifyRouteChange(tabId: number, url: string) {
  if (url === lastUrl) {
    return // Skip if URL hasn't changed
  }

  lastUrl = url
  chrome.tabs.sendMessage(
    tabId,
    { type: "ROUTE_CHANGED", url } as RouteChangeMessage,
    // Chrome handles connection errors automatically
    () => chrome.runtime.lastError // Acknowledge any error
  )
}

// Listen for tab updates (handles both regular navigation and SPA changes)
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  // Only process when URL changes and page is complete
  if (changeInfo.url || (changeInfo.status === "complete" && tab.url)) {
    const url = changeInfo.url || tab.url
    if (url?.includes("claude.ai")) {
      notifyRouteChange(tabId, url)
    }
  }
})

// Listen for history state updates (catches some SPA navigations that tabs API might miss)
chrome.webNavigation.onHistoryStateUpdated.addListener((details) => {
  // Only handle main frame
  if (details.frameId === 0 && details.url.includes("claude.ai")) {
    notifyRouteChange(details.tabId, details.url)
  }
})

// Log that service worker has started
console.log("[TC] Background service worker started")
