import React from "react"

import { MESSAGE_SOURCE, MESSAGE_TYPE } from "@/constants/constants"
import { MessageType } from "@/types"
import { normalizeContent } from "@/utils/format"

import { ModelInstruction } from "@/components/instruction-selector"

interface ContentMessage {
  source: typeof MESSAGE_SOURCE
  type: MessageType
  content?: string
}

interface ContentSyncOptions {
  instructions: ModelInstruction[] | undefined
  onValueChange: (value: string) => void
  onKeyChange: () => void
}

/**
 * Hook to sync content with window messages and match against instructions
 * @param options - Configuration options for content syncing
 */
export const useContentSync = ({
  instructions,
  onValueChange,
  onKeyChange,
}: ContentSyncOptions): void => {
  React.useEffect(() => {
    const handleMessage = (event: MessageEvent<ContentMessage>) => {
      try {
        // Validate message source
        if (event.source !== window || event.data.source !== MESSAGE_SOURCE) {
          return
        }

        // Handle content change messages
        if (event.data.type === MESSAGE_TYPE.CONTENT_CHANGED) {
          if (!event.data.content) {
            onValueChange("")
            onKeyChange()
            return
          }

          const normalizedContent = normalizeContent(event.data.content)
          const matchingInstruction = instructions?.find((instruction) => {
            const normalizedInstruction = normalizeContent(instruction.content)
            return normalizedContent === normalizedInstruction
          })

          if (matchingInstruction) {
            console.log(
              "🚀 [TC] ~ matchingInstruction:",
              matchingInstruction.label
            )
            onValueChange(matchingInstruction.value)
          } else {
            onValueChange("")
            onKeyChange()
          }
        }
      } catch (error) {
        console.error("[TC] Error processing content sync message:", error)
        onValueChange("")
        onKeyChange()
      }
    }

    window.addEventListener("message", handleMessage as EventListener)
    return () =>
      window.removeEventListener("message", handleMessage as EventListener)
  }, [instructions, onValueChange, onKeyChange])
}
