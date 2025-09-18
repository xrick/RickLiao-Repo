export function setupControls(
  control: HTMLElement,
  thinkingBlock: Element,
  resContainer: Element
) {
  const copyButton = control.querySelector("button")
  if (!copyButton) return

  copyButton.classList.add("tc-select-none")

  copyButton.addEventListener(
    "click",
    async (e) => {
      e.stopPropagation()
      await handleCopyClick(copyButton, thinkingBlock)
    },
    true
  )

  control.addEventListener("click", () => {
    const currentState = resContainer.getAttribute(
      "data-tc-thinking-block-state"
    )

    // Update the collapse state of thinking block in the response container
    const newState = currentState === "expanded" ? "collapsed" : "expanded"
    resContainer.setAttribute("data-tc-thinking-block-state", newState)

    // Toggle the collapse state of the thinking block as fallback
    thinkingBlock.classList.toggle("collapsed")
  })
}

async function handleCopyClick(copyButton: Element, codeBlock: Element) {
  const codeElement = codeBlock.querySelector("code")
  if (!codeElement) return

  try {
    await navigator.clipboard.writeText(codeElement.textContent || "")
    updateCopyButtonUI(copyButton)
  } catch (err) {
    console.error("[TC] Failed to copy:", err)
  }
}

function updateCopyButtonUI(copyButton: Element) {
  const textSpan = copyButton.querySelector("span")
  const svg = copyButton.querySelector("svg")
  if (!textSpan || !svg) return

  const originalText = textSpan.textContent
  const originalSvgPath = svg.innerHTML

  // textSpan.textContent = "Copied"

  svg.innerHTML =
    '<path d="M173.66,98.34a8,8,0,0,1,0,11.32l-56,56a8,8,0,0,1-11.32,0l-24-24a8,8,0,0,1,11.32-11.32L112,148.69l50.34-50.35A8,8,0,0,1,173.66,98.34ZM232,128A104,104,0,1,1,128,24,104.11,104.11,0,0,1,232,128Zm-16,0a88,88,0,1,0-88,88A88.1,88.1,0,0,0,216,128Z"></path>'

  setTimeout(() => {
    textSpan.textContent = originalText
    svg.innerHTML = originalSvgPath
  }, 2000)
}
