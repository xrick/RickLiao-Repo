export const formatLabel = (filename: string): string =>
  filename.replace(".md", "")

export const formatStarCount = (count: number): string => {
  if (count >= 1000000) {
    return (count / 1000000).toFixed(1).replace(/\.0$/, "") + "M"
  }
  if (count >= 1000) {
    return (count / 1000).toFixed(1).replace(/\.0$/, "") + "K"
  }
  return count.toString()
}

/**
 * Normalizes content by removing extra whitespace and trimming
 * @param content - The content string to normalize
 * @returns Normalized content string
 */
export const normalizeContent = (content: string): string => {
  return content.replace(/\s+/g, " ").trim()
}
