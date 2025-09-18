import { cn } from "@/lib/utils"

import type { ModelInstruction } from "."

interface InstructionDescriptionProps {
  error: string | null
  selectedInstruction?: ModelInstruction
}

const convertMarkdownToHtml = (markdown: string): string => {
  return markdown
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>") // Bold
    .replace(/\*(.*?)\*/g, "<em>$1</em>") // Italic
    .replace(/\n/g, "<br />") // New lines
}

export const InstructionDescription = ({
  error,
  selectedInstruction,
}: InstructionDescriptionProps) => (
  <div className="flex-1 mr-1 mb-1 text-wrap py-3 px-3.5 gap-2.5 rounded-lg border-0.5 transition max-sm:hidden bg-bg-100/40 border-border-300 text-text-200 tc-font-light tc-text-balance">
    <div
      className={cn(
        "font-tiempos text-[0.9375rem] leading-snug",
        !error && "tc-prose tc-prose-sm tc-prose-neutral dark:tc-prose-invert",
        !selectedInstruction?.description &&
          "tc-flex tc-items-center tc-justify-center tc-size-full tc-text-center"
      )}
    >
      {error ? (
        <span className="text-danger-100">{error}</span>
      ) : (
        <div
          key={selectedInstruction?.value} // Force re-render on instruction change
          className="tc-animate-fade-in"
          dangerouslySetInnerHTML={{
            __html: convertMarkdownToHtml(
              selectedInstruction?.description ||
                "Select a model instruction version to let Claude think"
            ),
          }}
        />
      )}
    </div>
  </div>
)
