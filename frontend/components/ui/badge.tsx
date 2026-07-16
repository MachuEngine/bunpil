import { cn } from "@/lib/utils";

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: "default" | "mc" | "sa" | "hard" | "med" | "easy" | "approved" | "rejected";
}

const variantStyles: Record<NonNullable<BadgeProps["variant"]>, string> = {
  default: "bg-[#E7EDE8] text-[#6E7469]",
  mc: "bg-[#E7EDE8] text-[#22392E]",
  sa: "bg-[#EFEAD9] text-[#6B5A2E]",
  hard: "bg-[#F7E9E4] text-[#A63B2E]",
  med: "bg-[#F5EBD8] text-[#93601F]",
  easy: "bg-[#E5EEE4] text-[#3D6B4A]",
  approved: "bg-[#E5EEE4] text-[#3D6B4A]",
  rejected: "bg-[#F7E9E4] text-[#A63B2E]",
};

export function Badge({ className, variant = "default", children, ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium",
        variantStyles[variant],
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
}
