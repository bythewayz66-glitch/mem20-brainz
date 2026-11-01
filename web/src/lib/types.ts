/**
 * Wire types for the mem20-brainz live stream.
 *
 * These mirror `src/mem20_brainz/server/frames.py` and `runner.py`. If the
 * backend schema changes, FRAME_SCHEMA_VERSION changes and this file must
 * follow — the app refuses to render a frame whose version it does not know.
 */

export const EXPECTED_SCHEMA_VERSION = 1;

export interface ReadoutInputs {
  [key: string]: number;
}

export interface Readout {
  name: string;
  value: number;
  unit: string;
  inputs: ReadoutInputs;
  formula: string;
  /** Always "derived_proxy". Rendered as a badge, never hidden. */
  provenance: string;
  interpretation: string;
}

export interface RegionActivity {
  name: string;
  n_neurons: number;
  rate_hz: number;
  active_fraction: number;
  mean_v_mv: number;
}

export interface FrameStatus {
  backend: string;
  engine: string;
  n_neurons: number;
  n_synapses: number;
  running: boolean;
  data_kind: string;
  neuron_model_tier: string;
  uptime_s: number;
  steps: number;
  total_spikes: number;
  steps_per_sec: number;
  frames_emitted: number;
}

export interface Provenance {
  kind: string;
  generator?: string;
  generator_version?: string;
  seed?: number;
  fitted_to?: string | null;
  fitted_to_note?: string;
  region_sizes?: Record<string, number>;
  n_neurons?: number;
  n_synapses?: number;
  neuron_model_tier?: string;
  parameters_are_placeholders?: boolean;
}

export interface Frame {
  schema_version: number;
  step: number;
  t_ms: number;
  dt_ms: number;
  v_q: number[];
  v_range: [number, number];
  v_bin_size: number;
  spikes: number[];
  regions: RegionActivity[];
  readouts: Readout[];
  status: FrameStatus;
  provenance: Provenance;
}

export interface RegionInfo {
  name: string;
  n_neurons: number;
  kind: string;
  center: [number, number];
}

export interface RunConfig {
  dt_ms: number;
  steps_per_frame: number;
  target_fps: number;
  baseline_rate_hz: number;
  threat_region: string;
  seed: number;
  tonic_current: number;
}

export interface Snapshot {
  schema_version: number;
  config: RunConfig;
  status: FrameStatus;
  provenance: Provenance;
  regions: RegionInfo[];
  /** Flat [x0, y0, x1, y1, ...] in [0,1]^2. Display coordinates only. */
  positions: number[][];
  readout_provenance: string;
  readout_disclaimer: string;
}

export type ConnectionState = "connecting" | "open" | "closed" | "error";

/** Region colours. Chosen for contrast on a dark base, not for meaning. */
export const REGION_COLORS: Record<string, string> = {
  sensory: "#4cc9f0",
  assoc_a: "#b5179e",
  assoc_b: "#7209b7",
  threat: "#f72585",
  inhib: "#90be6d",
};

export function regionColor(name: string): string {
  return REGION_COLORS[name] ?? "#8d99ae";
}
