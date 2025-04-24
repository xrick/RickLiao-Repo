type ObserverCallback = () => void

export interface MutationObserverOptions {
  childList?: boolean
  subtree?: boolean
  attributes?: boolean
  characterData?: boolean
  debounceTime?: number
}

export class MutationObserverService {
  private observer: MutationObserver | null = null
  private callbacks: Set<ObserverCallback> = new Set()
  private timeouts: Map<ObserverCallback, NodeJS.Timeout> = new Map()
  private isProcessing = false
  private options: MutationObserverOptions

  constructor(
    options: MutationObserverOptions = {
      childList: true,
      subtree: true,
      debounceTime: 200,
    }
  ) {
    this.options = options
  }

  initialize() {
    if (this.observer) return

    this.observer = new MutationObserver(() => {
      if (this.isProcessing) return

      this.isProcessing = true
      this.callbacks.forEach((callback) => {
        const existingTimeout = this.timeouts.get(callback)
        if (existingTimeout) {
          clearTimeout(existingTimeout)
        }

        const timeout = setTimeout(() => {
          callback()
          this.isProcessing = false
        }, this.options.debounceTime)

        this.timeouts.set(callback, timeout)
      })
    })

    this.observer.observe(document.body, {
      childList: this.options.childList,
      subtree: this.options.subtree,
      attributes: this.options.attributes,
      characterData: this.options.characterData,
    })
  }

  /* service-level cleanup for cleaning the observer */
  cleanup() {
    // 1. Disconnect the MutationObserver
    this.observer?.disconnect()
    // 2. Clear the observer reference
    this.observer = null
    // 3. Clear all pending timeouts
    this.timeouts.forEach((timeout) => clearTimeout(timeout))
    this.timeouts.clear()
    // 4. Clear all callbacks
    this.callbacks.clear()
    // 5. Reset processing flag
    this.isProcessing = false
  }

  subscribe(callback: ObserverCallback) {
    this.callbacks.add(callback)
    return () => this.unsubscribe(callback)
  }

  private unsubscribe(callback: ObserverCallback) {
    this.callbacks.delete(callback)
    const timeout = this.timeouts.get(callback)
    if (timeout) {
      clearTimeout(timeout)
      this.timeouts.delete(callback)
    }
  }
}
