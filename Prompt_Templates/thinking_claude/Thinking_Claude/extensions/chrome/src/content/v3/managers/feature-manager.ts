import { Feature } from "@/types"

export class FeatureManager {
  private features = new Map<string, Feature>()
  private cleanupFunctions = new Map<string, () => void>()

  /**
   * Register a new feature
   * @param feature Feature instance to register
   * @throws Error if feature with same id already exists
   */
  register(feature: Feature): void {
    if (this.features.has(feature.id)) {
      throw new Error(`[TC] Feature with id ${feature.id} already exists`)
    }
    this.features.set(feature.id, feature)
  }

  /**
   * Initialize all registered features
   */
  initialize(): void {
    this.features.forEach((feature, id) => {
      try {
        const cleanup = feature.initialize()
        if (cleanup) {
          this.cleanupFunctions.set(id, cleanup)
        }
      } catch (error) {
        console.error(`[TC] Failed to initialize feature ${id}:`, error)
      }
    })
  }

  /**
   * Clean up all features
   */
  cleanup(): void {
    this.cleanupFunctions.forEach((cleanup, id) => {
      try {
        cleanup()
      } catch (error) {
        console.error(`[TC] Failed to cleanup feature ${id}:`, error)
      }
    })
    this.cleanupFunctions.clear()
    this.features.clear()
  }

  /**
   * Get a registered feature by id
   */
  getFeature(id: string): Feature | undefined {
    return this.features.get(id)
  }
}
