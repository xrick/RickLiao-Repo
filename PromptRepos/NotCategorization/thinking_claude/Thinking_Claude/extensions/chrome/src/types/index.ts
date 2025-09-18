import { MESSAGE_TYPE } from "@/constants/constants"

/**
 * Base interface for all features
 */
export interface Feature {
  id: string
  initialize(): void | (() => void) // Return cleanup function if needed
}

export type MessageType = (typeof MESSAGE_TYPE)[keyof typeof MESSAGE_TYPE]
