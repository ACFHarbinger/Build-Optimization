export function StatusPill({ children, kind = "info" }: { children: React.ReactNode; kind?: "good" | "warning" | "error" | "info" }) {
  return <span className={`status-pill ${kind}`}>{children}</span>;
}
