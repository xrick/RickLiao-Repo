import type { MutationObserverService } from "@/services/mutation-observer"
import { shouldInitialize } from "@/utils/url-utils"

import { BaseFeature } from "../base-feature"
import { processThinkingBlocks } from "./process-thinking-block"

/**
 * Feature that adds toggle functionality to thinking blocks in the UI
 * Manages the collapse/expand and copy functionality for code blocks
 */
export class TCThinkingBlock extends BaseFeature {
  /**
   * @param mutationObserver - Service to observe DOM changes for thinking blocks
   */
  constructor(private mutationObserver: MutationObserverService) {
    super("tc-thinking-block")
  }

  /**
   * Initialize the thinking block feature
   * Sets up mutation observer to watch for new thinking blocks
   * @returns Cleanup function to unsubscribe from mutation observer and remove custom attributes
   */
  initialize(): void | (() => void) {
    if (!shouldInitialize(window.location.href)) {
      return
    }

    this.mutationObserver.initialize()

    const unsubscribe = this.mutationObserver.subscribe(processThinkingBlocks)

    return () => {
      // Unsubscribe from mutation observer
      unsubscribe()

      // Clean up all feature-specific attributes
      document
        .querySelectorAll("[data-tc-processed]")
        .forEach((el) => el.removeAttribute("data-tc-processed"))
      document
        .querySelectorAll("[data-tc-thinking-block-state]")
        .forEach((el) => el.removeAttribute("data-tc-thinking-block-state"))
    }
  }
}
