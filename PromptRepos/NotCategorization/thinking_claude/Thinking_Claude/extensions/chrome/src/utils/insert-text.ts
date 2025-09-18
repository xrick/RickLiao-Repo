import { CLAUDE_INPUT_TEXTAREA } from "@/selectors/input-selectors"

export const insertTextIntoClaudeInput = async (
  content: string | null = ""
): Promise<boolean> => {
  const maxRetries = 3
  const retryDelay = 100 // ms

  const insertWithClipboard = async () => {
    try {
      // Store original clipboard content
      const originalClipboard = await navigator.clipboard
        .readText()
        .catch(() => null)

      // Copy new content to clipboard
      await navigator.clipboard.writeText(content ?? "")

      // Get the contenteditable div
      const inputDiv = document.querySelector(
        CLAUDE_INPUT_TEXTAREA
      ) as HTMLDivElement

      if (!inputDiv) return false

      // Focus and clear the input
      inputDiv.focus()
      inputDiv.textContent = ""

      // Paste content using keyboard event
      document.execCommand("paste")

      // Dispatch input event
      const inputEvent = new InputEvent("input", {
        bubbles: true,
        cancelable: true,
        inputType: "insertFromPaste",
      })
      inputDiv.dispatchEvent(inputEvent)

      // Restore original clipboard content
      if (originalClipboard !== null) {
        await navigator.clipboard.writeText(originalClipboard)
      }

      // Verify insertion
      return inputDiv.textContent?.trim() === (content ?? "").trim()
    } catch (error) {
      console.error("[TC] Failed to insert text:", error)
      return false
    }
  }

  // Retry logic
  for (let i = 0; i < maxRetries; i++) {
    const success = await insertWithClipboard()
    if (success) return true

    if (i < maxRetries - 1) {
      await new Promise((resolve) => setTimeout(resolve, retryDelay))
    }
  }

  // Fallback to direct textContent setting if clipboard method fails
  try {
    const inputDiv = document.querySelector(
      CLAUDE_INPUT_TEXTAREA
    ) as HTMLDivElement

    if (!inputDiv) return false

    inputDiv.textContent = content ?? ""
    inputDiv.focus()

    const inputEvent = new InputEvent("input", {
      bubbles: true,
      cancelable: true,
      inputType: "insertText",
      data: content,
    })
    inputDiv.dispatchEvent(inputEvent)

    return true
  } catch (error) {
    console.error("[TC] Fallback insertion failed:", error)
    return false
  }
}
