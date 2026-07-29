import { create } from "zustand";

interface AppState {
  autoRefresh: boolean;
  refreshInterval: number;
  refreshNonce: number;
  setAutoRefresh: (value: boolean) => void;
  setRefreshInterval: (value: number) => void;
  bumpRefresh: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  autoRefresh: false,
  refreshInterval: 5,
  refreshNonce: 0,
  setAutoRefresh: (value) => set({ autoRefresh: value }),
  setRefreshInterval: (value) => set({ refreshInterval: value }),
  bumpRefresh: () => set((s) => ({ refreshNonce: s.refreshNonce + 1 })),
}));
