export const CLAUDE_INPUT_CONTAINER = "fieldset > div.cursor-text"

// Target the outermost flex container that holds all buttons
export const CLAUDE_INPUT_BUTTONS_CONTAINER = `${CLAUDE_INPUT_CONTAINER}  div.flex.min-w-0.min-h-4.flex-1.items-center`

// Target the input content area
export const CLAUDE_INPUT_TEXTAREA =
  'div.ProseMirror.break-words[contenteditable="true"][translate="no"]'

// Target the paragraph containing user input
export const CLAUDE_INPUT_CONTENT = `${CLAUDE_INPUT_TEXTAREA} > p`
