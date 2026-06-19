import { cn } from "@/lib/utils";

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: "default" | "mc" | "sa" | "hard" | "med" | "easy" | "approved" | "rejected";
}

const variantStyles: Record<NonNullable<BadgeProps["variant"]>, string> = {
  default: "bg-[#F0EEE9] text-[#6B6B6B]",
  mc: "bg-blue-100 text-blue-700",
  sa: "bg-purple-100 text-purple-700",
  hard: "bg-red-100 text-red-700",
  med: "bg-orange-100 text-orange-700",
  easy: "bg-green-100 text-green-700",
  approved: "bg-green-100 text-green-700",
  rejected: "bg-red-100 text-red-700",
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
