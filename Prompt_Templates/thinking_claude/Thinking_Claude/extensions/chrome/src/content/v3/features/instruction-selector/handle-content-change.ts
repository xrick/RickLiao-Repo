import { MESSAGE_SOURCE, MESSAGE_TYPE } from "@/constants/constants"
import { CLAUDE_INPUT_TEXTAREA } from "@/selectors"
import { MessageType } from "@/types"
import { normalizeContent } from "@/utils/format"

interface ContentMessage {
  source: typeof MESSAGE_SOURCE
  type: MessageType
  content: string | null
}

// Track last content to prevent unnecessary updates
let lastContent: string | null = null
let debounceTimer: number | undefined

/**
 * Posts a content change message to the window
 * @param content The content to send, or null if no content
 */
function postContentMessage(content: string | null) {
  // Clear any pending debounce
  if (debounceTimer) {
    window.clearTimeout(debounceTimer)
  }

  // Only send if content actually changed
  if (content !== lastContent) {
    debounceTimer = window.setTimeout(() => {
      const message: ContentMessage = {
        source: MESSAGE_SOURCE,
        type: MESSAGE_TYPE.CONTENT_CHANGED,
        content,
      }
      window.postMessage(message, "*")
      lastContent = content
    }, 100) // Debounce for 100ms
  }
}

/**
 * Handles content changes in the Claude input textarea
 * @param hasAttribute Whether the input container has been initialized
 */
export function handleContentChange(hasAttribute: boolean) {
  if (!hasAttribute) return

  const inputDiv = document.querySelector(CLAUDE_INPUT_TEXTAREA)
  if (!inputDiv) {
    postContentMessage(null)
    return
  }

  const contentP = inputDiv.querySelector("p")
  if (!contentP) {
    postContentMessage(null)
    return
  }

  const currentContent = contentP.textContent?.trim() || ""
  const normalizedContent = normalizeContent(currentContent)

  // Send normalized content
  postContentMessage(normalizedContent || null)
}
