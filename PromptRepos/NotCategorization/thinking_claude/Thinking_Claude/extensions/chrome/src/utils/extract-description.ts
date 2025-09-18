export const extractDescription = (content: string): string => {
  // Look for content between <anthropic_thinking_protocol> tags, including newlines
  const protocolMatch = content.match(
    /<anthropic_thinking_protocol>\s*([\s\S]*?)\s*(?:<\/anthropic_thinking_protocol>|$)/
  )

  if (protocolMatch && protocolMatch[1]) {
    const protocolContent = protocolMatch[1].trim()
    // Remove any extra newlines and whitespace
    const cleanContent = protocolContent.replace(/\s+/g, " ")
    return cleanContent.length > 260
      ? cleanContent.substring(0, 260) + "..."
      : cleanContent
  }

  return "Model instruction"
}
