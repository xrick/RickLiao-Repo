import { Feature } from "@/types"

/**
 * Base abstract class for features
 * Provides common functionality and enforces feature contract
 */
export abstract class BaseFeature implements Feature {
  constructor(readonly id: string) {}

  /**
   * Initialize the feature
   * @returns cleanup function if needed
   */
  abstract initialize(): void | (() => void)

  /**
   * Helper method to safely add event listeners with automatic cleanup
   */
  protected addEventListenerWithCleanup(
    element: Element,
    event: string,
    handler: EventListener,
    options?: boolean | AddEventListenerOptions
  ): () => void {
    element.addEventListener(event, handler, options)
    return () => element.removeEventListener(event, handler, options)
  }
}
