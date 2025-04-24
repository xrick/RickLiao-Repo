import React from "react"

import { GITHUB_API_URL } from "@/constants/constants"
import { extractDescription } from "@/utils/extract-description"
import { formatLabel } from "@/utils/format"
import { insertTextIntoClaudeInput } from "@/utils/insert-text"

import { ModelInstruction } from "@/components/instruction-selector"

interface GitHubFile {
  name: string
  path: string
  sha: string
  download_url: string
}

export const useModelInstructions = () => {
  const [instructions, setInstructions] = React.useState<ModelInstruction[]>([])
  const [isLoading, setIsLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)
  const [starsCount, setStarsCount] = React.useState<number | null>(null)
  const abortControllerRef = React.useRef<AbortController | null>(null)

  const fetchModelInstructions = async (signal: AbortSignal) => {
    try {
      setError(null)

      const response = await fetch(GITHUB_API_URL, { signal })
      if (!response.ok) {
        throw new Error(
          `Failed to fetch model instructions: ${response.status}`
        )
      }
      const files: GitHubFile[] = await response.json()

      const instructionFiles = files.filter(
        (file) => file.name !== "changelog.md"
      )

      // Process in batches to avoid overwhelming the network
      const BATCH_SIZE = 3
      const modelInstructions: ModelInstruction[] = []

      for (let i = 0; i < instructionFiles.length; i += BATCH_SIZE) {
        const batch = instructionFiles.slice(i, i + BATCH_SIZE)
        const batchResults = await Promise.all(
          batch.map(async (file) => {
            try {
              const contentResponse = await fetch(file.download_url, {
                signal,
              })
              if (!contentResponse.ok) {
                throw new Error(
                  `Failed to fetch content for ${file.name}: ${contentResponse.status}`
                )
              }
              const content = await contentResponse.text()

              return {
                value: file.name,
                label: formatLabel(file.name),
                description: extractDescription(content),
                content: content,
              }
            } catch (error) {
              console.error(`Error processing ${file.name}:`, error)
              // Return a placeholder for failed items instead of breaking the whole batch
              return {
                value: file.name,
                label: formatLabel(file.name),
                description: "Failed to load instruction",
                content: "",
              }
            }
          })
        )
        modelInstructions.push(...batchResults)

        // Small delay between batches to prevent rate limiting
        if (i + BATCH_SIZE < instructionFiles.length) {
          await new Promise((resolve) => setTimeout(resolve, 100))
        }
      }

      const sortedInstructions = [...modelInstructions].reverse()
      setInstructions(sortedInstructions)
    } catch (err) {
      // Ignore aborted requests
      if (err instanceof Error && err.name === "AbortError") {
        return
      }
      console.error("Error fetching instructions:", err)
      setError(
        err instanceof Error ? err.message : "Failed to load model instructions"
      )
      setInstructions([])
    }
  }

  const fetchStarsCount = async (signal: AbortSignal) => {
    try {
      const response = await fetch(
        "https://api.github.com/repos/richards199999/Thinking-Claude",
        { signal }
      )
      if (!response.ok) {
        throw new Error(`Failed to fetch stars count: ${response.status}`)
      }
      const data = await response.json()
      setStarsCount(data.stargazers_count)
    } catch (err) {
      if (err instanceof Error && err.name === "AbortError") {
        return
      }
      console.error("Error fetching stars count:", err)
    }
  }

  const handleInstructionSelect = async (instruction: ModelInstruction) => {
    if (instruction.content) {
      await insertTextIntoClaudeInput(instruction.content)
    }
  }

  React.useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true)
      setError(null)

      // Create new AbortController for this fetch
      abortControllerRef.current = new AbortController()
      const { signal } = abortControllerRef.current

      try {
        // Fetch both stars and instructions in parallel
        await Promise.all([
          fetchStarsCount(signal),
          fetchModelInstructions(signal),
        ])
      } catch (err) {
        // Handle any errors from the parallel fetches
        if (err instanceof Error && err.name !== "AbortError") {
          console.error("Error during parallel fetches:", err)
        }
      } finally {
        setIsLoading(false)
      }
    }

    fetchData()

    return () => {
      abortControllerRef.current?.abort()
    }
  }, [])

  return {
    instructions,
    isLoading,
    error,
    starsCount,
    handleInstructionSelect,
  }
}
