import { CLAUDE_INPUT_CONTAINER } from "@/selectors/input-selectors"

import { handleContentChange } from "./handle-content-change"
import { cleanupSelect, handleSelectComponent } from "./handle-select-component"

export function processInputContainer() {
  const inputContainer = document.querySelector(CLAUDE_INPUT_CONTAINER)
  if (!inputContainer) return

  handleContentChange(inputContainer.hasAttribute("data-tc-input-container"))

  if (inputContainer.hasAttribute("data-tc-input-container")) return

  inputContainer.setAttribute("data-tc-input-container", "true")

  // Handle select component initialization
  handleSelectComponent()
}

// Cleanup function for the feature
export function cleanupInputContainer() {
  // Clean up React root and interval
  cleanupSelect()

  // Clean up attributes
  document
    .querySelectorAll("[data-tc-input-container]")
    .forEach((el) => el.removeAttribute("data-tc-input-container"))
}
