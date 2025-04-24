import { THINKING_BLOCK_CONTROLS_SELECTORS } from "@/selectors"

import { setupControls } from "./setup-controls"

export function processThinkingBlocks() {
  const thinkingBlockControls = document.querySelectorAll(
    THINKING_BLOCK_CONTROLS_SELECTORS
  )

  thinkingBlockControls.forEach(processControl)
}

function processControl(control: Element) {
  if (control.hasAttribute("data-tc-processed")) return
  control.setAttribute("data-tc-processed", "true")

  const resContainer = control.closest("div[data-is-streaming]") as HTMLElement
  const thinkingBlock = control
    .closest("pre")
    ?.querySelector(".code-block__code")
  if (!thinkingBlock) return

  if (!resContainer.hasAttribute("data-tc-thinking-block-state")) {
    resContainer.setAttribute("data-tc-thinking-block-state", "expanded")
  }

  setupControls(control as HTMLElement, thinkingBlock, resContainer)
}
