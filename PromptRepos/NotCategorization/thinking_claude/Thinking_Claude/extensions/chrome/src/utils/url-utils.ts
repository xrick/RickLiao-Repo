export const isChatPage = (url: string): boolean => {
  return url.startsWith("https://claude.ai/chat/")
}

export const isNewChatPage = (url: string): boolean => {
  return url.startsWith("https://claude.ai/new")
}

export const shouldInitialize = (url: string, page?: string): boolean => {
  return page === "new" ? isNewChatPage(url) : isChatPage(url)
}
