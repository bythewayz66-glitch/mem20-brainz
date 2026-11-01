/** Thin REST client for the control endpoints. */

import type { Frame, RunConfig, Snapshot } from "./types";

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${path} failed (${res.status}): ${detail}`);
  }
  return (await res.json()) as T;
}

export const api = {
  start: () => post<{ ok: boolean }>("/api/run/start"),
  pause: () => post<{ ok: boolean }>("/api/run/pause"),
  reset: () => post<{ ok: boolean; state: Snapshot }>("/api/run/reset"),
  step: () => post<Frame>("/api/run/step"),
  setConfig: (update: Partial<RunConfig>) =>
    post<{ ok: boolean; config: RunConfig }>("/api/config", update),
  stimulate: (region: string, strength: number, duration_ms: number) =>
    post<{ ok: boolean }>("/api/stimulus", { region, strength, duration_ms }),
};
