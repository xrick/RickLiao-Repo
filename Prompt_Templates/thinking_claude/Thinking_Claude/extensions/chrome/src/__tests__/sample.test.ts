import { beforeEach, describe, expect, it, vi, type Mock } from "vitest"

interface ChromeMock {
  runtime: {
    sendMessage: Mock
    onMessage: {
      addListener: Mock
    }
  }
  storage: {
    local: {
      get: Mock
      set: Mock
    }
  }
}

// Mock chrome API
const mockChrome = {
  runtime: {
    sendMessage: vi.fn(),
    onMessage: {
      addListener: vi.fn(),
    },
  },
  storage: {
    local: {
      get: vi.fn().mockImplementation(() => Promise.resolve({})),
      set: vi.fn().mockImplementation(() => Promise.resolve()),
    },
  },
} as unknown as ChromeMock

// Add chrome to global
;(globalThis as any).chrome = mockChrome

describe("Chrome Extension Basic Tests", () => {
  beforeEach(() => {
    // Clear all mocks before each test
    vi.clearAllMocks()
  })

  it("should send messages correctly", () => {
    const message = { type: "TEST_MESSAGE", data: "test data" }
    chrome.runtime.sendMessage(message)

    expect(mockChrome.runtime.sendMessage).toHaveBeenCalledWith(message)
  })

  it("should handle storage operations", async () => {
    const testData = { key: "value" }

    // Setup the mock to return our test data
    mockChrome.storage.local.get.mockImplementationOnce(() =>
      Promise.resolve(testData)
    )

    const result = await chrome.storage.local.get("key")
    expect(result).toEqual(testData)
  })

  it("should add message listeners", () => {
    const messageListener = (message: any) => console.log(message)
    chrome.runtime.onMessage.addListener(messageListener)

    expect(mockChrome.runtime.onMessage.addListener).toHaveBeenCalledWith(
      messageListener
    )
  })
})
