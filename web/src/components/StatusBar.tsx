/**
 * Status bar: what is actually running, and whether the data is measured.
 *
 * The provenance chip is not decorative. It reads `data_kind` from the frame,
 * so a viewer can always tell that this network is synthetic.
 */

import type { ConnectionState, FrameStatus, Provenance } from "../lib/types";

interface Props {
  state: ConnectionState;
  status: FrameStatus | null;
  provenance: Provenance | null;
  frameCount: number;
  error: string | null;
}

export function StatusBar({ state, status, provenance, frameCount, error }: Props) {
  const kind = status?.data_kind ?? provenance?.kind ?? "unknown";
  const isSynthetic = kind === "synthetic";

  return (
    <footer className="statusbar">
      <span className={`conn conn-${state}`}>
        <span className="dot" />
        {state === "open" ? "connected" : state}
      </span>

      {error && <span className="status-error">{error}</span>}

      <span className="status-item">
        backend <code>{status?.backend ?? "—"}</code>
      </span>
      <span className="status-item">
        engine <code>{status?.engine ?? "—"}</code>
      </span>
      <span className="status-item">
        neurons <code>{(status?.n_neurons ?? 0).toLocaleString()}</code>
      </span>
      <span className="status-item">
        synapses <code>{(status?.n_synapses ?? 0).toLocaleString()}</code>
      </span>
      <span className="status-item">
        steps/s <code>{(status?.steps_per_sec ?? 0).toFixed(0)}</code>
      </span>
      <span className="status-item">
        frames <code>{frameCount}</code>
      </span>
      <span className="status-item">
        tier <code>{status?.neuron_model_tier ?? "—"}</code>
      </span>

      <span className={`chip ${isSynthetic ? "chip-synth" : "chip-measured"}`}>
        data: {kind}
        {isSynthetic && " — not measured"}
      </span>
    </footer>
  );
}
