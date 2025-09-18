import { SelectItem } from "@/components/ui/select"

// Constants
const ITEM_STYLES =
  "py-1 px-2 rounded-md cursor-pointer whitespace-nowrap overflow-hidden text-ellipsis grid grid-cols-[minmax(0,_1fr)_auto] gap-2 items-center outline-none select-none [&[data-highlighted]]:bg-bg-300 [&[data-highlighted]]:text-text-000 pr-0 mb-0.5 line-clamp-2 leading-tight tc-text-base tc-w-full tc-pr-6"

export const InstructionItem = ({
  value,
  label,
}: {
  value: string
  label: string
}) => (
  <SelectItem value={value} className={ITEM_STYLES}>
    <div className="flex items-center justify-between">
      <div
        className="flex-1 tc-text-nowrap tc-overflow-hidden tc-text-ellipsis"
        title={label}
      >
        {label}
      </div>
    </div>
  </SelectItem>
)
