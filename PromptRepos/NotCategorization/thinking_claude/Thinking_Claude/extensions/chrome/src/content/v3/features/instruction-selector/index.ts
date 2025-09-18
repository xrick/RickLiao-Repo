import type { MutationObserverService } from "@/services/mutation-observer"
import { shouldInitialize } from "@/utils/url-utils"

import { BaseFeature } from "../base-feature"
import {
  cleanupInputContainer,
  processInputContainer,
} from "./process-input-container"

/**
 * Feature that adds input selection functionality
 */
export class TCInstructionSelector extends BaseFeature {
  /**
   * @param mutationObserver - Service to observe DOM changes for input
   */
  constructor(private mutationObserver: MutationObserverService) {
    super("tc-instruction-selector")
  }

  /**
   * Initialize the input selector feature
   * Sets up mutation observer for input container
   * @returns Cleanup function to unsubscribe from mutation observer and remove custom attributes
   */
  initialize(): void | (() => void) {
    if (!shouldInitialize(window.location.href, "new")) {
      return
    }

    this.mutationObserver.initialize()

    // Subscribe to both input container changes and content changes
    const unsubscribeContainer = this.mutationObserver.subscribe(
      processInputContainer
    )

    return () => {
      unsubscribeContainer()
      cleanupInputContainer()
    }
  }
}
