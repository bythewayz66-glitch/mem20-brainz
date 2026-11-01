/** Shown when the stream is not live. Never renders stale data as current. */

import type { ConnectionState } from "../lib/types";

interface Props {
  state: ConnectionState;
  error: string | null;
  onReconnect: () => void;
}

export function Disconnected({ state, error, onReconnect }: Props) {
  return (
    <div className="disconnected">
      <div className="disconnected-card">
        <h2>stream disconnected</h2>
        <p>
          The simulation stream is not live, so no data is being shown. The last frame has
          been discarded rather than left on screen — a frozen frame would misrepresent the
          state of the run.
        </p>
        {error && <p className="disconnected-error">{error}</p>}
        <p className="disconnected-hint">
          Start the backend with <code>python -m mem20_brainz.server</code> and the frontend
          with <code>npm run dev</code> in <code>web/</code>, or use{" "}
          <code>scripts/dev.sh</code> to start both.
        </p>
        <button type="button" className="btn btn-primary" onClick={onReconnect}>
          {state === "connecting" ? "connecting…" : "reconnect"}
        </button>
      </div>
    </div>
  );
}
