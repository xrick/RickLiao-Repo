import { MutationObserverService } from "@/services/mutation-observer"

import { TCInstructionSelector } from "../features/instruction-selector"
import { TCThinkingBlock } from "../features/thinking-block"
import { FeatureManager } from "./feature-manager"

/**
 * Manages the lifecycle and coordination of all extension features and services
 */
export class ExtensionManager {
  private featureManager: FeatureManager
  private defaultMutationObserver: MutationObserverService
  private inputObserver: MutationObserverService
  private isInitialized = false

  constructor() {
    this.defaultMutationObserver = new MutationObserverService()
    this.inputObserver = new MutationObserverService({
      childList: true,
      subtree: true,
      attributes: true, // Watch for attribute changes
      debounceTime: 500,
    })
    this.featureManager = new FeatureManager()
  }

  /**
   * Register all extension features
   */
  private registerFeatures(): void {
    // Register features with their required services
    this.featureManager.register(
      new TCThinkingBlock(this.defaultMutationObserver)
    )
    this.featureManager.register(new TCInstructionSelector(this.inputObserver))
  }

  /**
   * Initialize the extension if conditions are met
   */
  initialize(): void {
    if (this.isInitialized) {
      console.log("[TC] 🔄 Features already initialized, skipping...")
      return
    }

    const initializeFeatures = () => {
      this.registerFeatures()
      this.featureManager.initialize()
      this.isInitialized = true
    }

    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", initializeFeatures, {
        once: true,
      })
    } else {
      initializeFeatures()
    }
  }

  /**
   * Clean up all features and services
   */
  cleanup(): void {
    this.featureManager.cleanup()
    this.defaultMutationObserver.cleanup()
    this.inputObserver.cleanup()
    this.isInitialized = false
  }
}
