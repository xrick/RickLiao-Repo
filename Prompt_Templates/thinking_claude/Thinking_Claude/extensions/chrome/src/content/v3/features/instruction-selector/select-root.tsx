import React from "react"

import { createRoot } from "react-dom/client"

import { InstructionSelect } from "@/components/instruction-selector"

/**
 * Creates a root element and mounts the SelectDemo component
 */
export const createSelectRoot = (container: HTMLElement) => {
  // Create a container for our React app
  const rootElement = document.createElement("div")
  rootElement.className = "flex items-center min-w-0"

  // Simply append to the end of the container
  container.appendChild(rootElement)

  // Create React root and render
  const root = createRoot(rootElement)
  root.render(
    <React.StrictMode>
      <InstructionSelect />
    </React.StrictMode>
  )

  return () => {
    root.unmount()
    rootElement.remove()
  }
}
