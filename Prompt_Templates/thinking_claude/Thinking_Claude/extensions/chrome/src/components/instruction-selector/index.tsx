import * as React from "react"

import { formatStarCount } from "@/utils/format"
import { insertTextIntoClaudeInput } from "@/utils/insert-text"
import { GitHubLogoIcon, StarFilledIcon } from "@radix-ui/react-icons"

import { cn } from "@/lib/utils"

import { useModelInstructions } from "@/hooks/use-model-instructions"

import { Badge } from "@/components/ui/badge"
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

import { useContentSync } from "../../hooks/use-content-sync"
import { InstructionDescription } from "./instruction-description"
import { InstructionItem } from "./instruction-item"

// Types
export interface ModelInstruction {
  value: string
  label: string
  description?: string
  content: string
}

const LoadingState = ({ isLoading }: { isLoading: boolean }) => (
  <div
    className={cn(
      "tc-min-w-24 text-text-500 tc-text-xs tc-bg-clip-text tc-text-transparent tc-bg-[length:200%_100%]",
      isLoading &&
        "tc-animate-shimmer tc-bg-gradient-to-r tc-from-gray-400/70 tc-via-gray-300 tc-to-gray-400/70"
    )}
  >
    Loading model instructions...
  </div>
)

export function InstructionSelect() {
  const [value, setValue] = React.useState("")
  const [key, setKey] = React.useState(Date.now())
  const [hoveredInstruction, setHoveredInstruction] =
    React.useState<ModelInstruction | null>(null)
  const {
    instructions,
    isLoading,
    starsCount,
    handleInstructionSelect,
    error,
  } = useModelInstructions()

  useContentSync({
    instructions,
    onValueChange: setValue,
    onKeyChange: () => setKey(Date.now()),
  })

  const handleClear = async (e: React.MouseEvent) => {
    e.stopPropagation()
    setValue("")
    setKey(Date.now())
    await insertTextIntoClaudeInput("")
  }

  const handleInstructionClick = async (instruction: ModelInstruction) => {
    if (instruction.content) {
      setValue(instruction.value)
      await handleInstructionSelect(instruction)
    }
  }

  const selectedInstruction = instructions.find((inst) => inst.value === value)
  const displayedInstruction = hoveredInstruction || selectedInstruction

  return (
    <div className="tc-min-w-24">
      {isLoading || instructions.length === 0 ? (
        <LoadingState isLoading={true} />
      ) : (
        <Select
          key={key}
          value={value}
          onValueChange={(value) => {
            handleInstructionClick(
              instructions.find((inst) => inst.value === value)!
            )
          }}
        >
          <SelectTrigger className="inline-flex items-center justify-center relative shrink-0 ring-offset-2 ring-offset-bg-300 ring-accent-main-100 focus-visible:outline-none focus-visible:ring-1 tc-shadow-none disabled:pointer-events-none disabled:opacity-50 disabled:shadow-none disabled:drop-shadow-none max-w-full min-w-0 pl-1.5 pr-1 h-7 ml-0.5 mr-1 hover:bg-bg-200 hover:border-border-400 border-0.5 text-sm rounded-md border-transparent transition text-text-500 hover:text-text-200 font-tiempos !tc-font-normal tc-gap-x-1">
            <SelectValue placeholder="Let Claude think" />
          </SelectTrigger>
          <SelectContent
            style={{
              zIndex: 9999,
            }}
            className="z-50 bg-bg-200 backdrop-blur-xl border-0.5 border-border-300 rounded-xl min-w-[12rem] overflow-hidden p-1 text-text-200 shadow-[0_0_0_0.5px_rgba(0,0,0,0.1),0_0_20px_rgba(0,0,0,0.05),0_1px_5px_rgba(0,0,0,0.1)] w-64 sm:w-[28rem] md:tc-w-[32rem] !z-30"
          >
            <div className="sm:flex justify-between items-center flex-1 text-xs font-medium text-text-300 px-1.5 pt-1 pb-1.5 min-h-5">
              <div className="translate-y-[0.5px]">
                Which model instruction should Claude use?
              </div>
              <a
                href="https://github.com/richards199999/Thinking-Claude"
                target="_blank"
                rel="noopener noreferrer"
              >
                <Badge
                  variant="default"
                  className="border-0.5 border-border-300 tc-flex tc-items-center tc-gap-2 tc-cursor-pointer hover:!bg-accent-main-100 hover:!text-oncolor-100 hover:!border-transparent transition tc-font-normal tc-text-xs tc-flex-nowrap"
                >
                  <span
                    title="Open-souced on GitHub"
                    className="tc-flex tc-items-center tc-justify-center"
                  >
                    <GitHubLogoIcon className="tc-size-3" />
                  </span>
                  <span className="tc-flex tc-items-center tc-justify-center">
                    {starsCount && (
                      <span
                        className="tc-text-xs"
                        title={`${starsCount} stars`}
                      >
                        {formatStarCount(starsCount)}
                      </span>
                    )}
                    <StarFilledIcon className="tc-size-3" />
                  </span>
                </Badge>
              </a>
            </div>
            <div className="grid sm:grid-cols-2 tc-gap-2 mt-0.5 pb-1 px-1">
              <div className="min-h-0">
                <div className="overflow-x-visible overflow-y-auto scroll-pb-6 min-h-[0px] [scrollbar-color:hsl(var(--text-500))] scroll-smooth overscroll-contain [-webkit-overflow-scrolling:touch] [&::-webkit-scrollbar]:mt-4 [&::-webkit-scrollbar]:w-[0.25rem] [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-track]:my-1 [&::-webkit-scrollbar-thumb]:rounded-[1em] [&::-webkit-scrollbar-thumb]:border-[0.25rem] [&::-webkit-scrollbar-thumb]:border-transparent [&::-webkit-scrollbar-thumb]:bg-clip-padding [&::-webkit-scrollbar-thumb]:bg-text-500/80 [&::-webkit-scrollbar-thumb:hover]:bg-text-500 sm:mr-1 min-h-40 max-h-64">
                  <SelectGroup>
                    {instructions.map((instruction) => (
                      <div
                        key={instruction.value}
                        onMouseEnter={() => setHoveredInstruction(instruction)}
                        onMouseLeave={() => setHoveredInstruction(null)}
                      >
                        <InstructionItem
                          value={instruction.value}
                          label={instruction.label}
                        />
                      </div>
                    ))}
                    <button
                      onClick={handleClear}
                      className="py-1 px-2 rounded-md cursor-pointer whitespace-nowrap overflow-hidden text-ellipsis grid grid-cols-[minmax(0,_1fr)_auto] gap-2 items-center outline-none select-none [&[data-highlighted]]:bg-bg-300 [&[data-highlighted]]:text-text-000 bg-transparent border-0.5 border-border-300 hover:!bg-accent-main-100 hover:!text-oncolor-100 hover:!border-transparent transition mb-1 mt-4 !rounded-lg text-center text-sm font-medium w-full"
                    >
                      Clear selection
                    </button>
                  </SelectGroup>
                </div>
              </div>
              <div className="flex flex-col">
                <InstructionDescription
                  error={error}
                  selectedInstruction={displayedInstruction}
                />
              </div>
            </div>
          </SelectContent>
        </Select>
      )}
    </div>
  )
}
