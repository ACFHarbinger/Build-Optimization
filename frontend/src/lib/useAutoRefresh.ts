import { useEffect, useRef } from "react";
import { useAppStore } from "../store/useAppStore";

/** Re-runs `callback` on the sidebar's auto-refresh interval when enabled. */
export function useAutoRefresh(callback: () => void) {
  const autoRefresh = useAppStore((s) => s.autoRefresh);
  const refreshInterval = useAppStore((s) => s.refreshInterval);

  const callbackRef = useRef(callback);
  callbackRef.current = callback;

  useEffect(() => {
    if (!autoRefresh) return;
    const id = setInterval(() => callbackRef.current(), refreshInterval * 1000);
    return () => clearInterval(id);
  }, [autoRefresh, refreshInterval]);
}
