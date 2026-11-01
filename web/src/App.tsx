import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./lib/api";
import { useLiveStream } from "./lib/useLiveStream";
import type { RunConfig } from "./lib/types";
import { EXPECTED_SCHEMA_VERSION } from "./lib/types";
import { NeuronCanvas } from "./components/NeuronCanvas";
import { Timeline, toTimelinePoint, type TimelinePoint } from "./components/Timeline";
import { ReadoutPanel } from "./components/ReadoutPanel";
import { Controls } from "./components/Controls";
import { StatusBar } from "./components/StatusBar";
import { Disconnected } from "./components/Disconnected";

/** How many frames of history the timeline keeps. */
const HISTORY_LIMIT = 240;

const TRACE_NAMES = ["arousal", "valence", "surprise", "confusion", "fear_like"];

export default function App() {
  const { state, snapshot, frame, frameCount, error, reconnect } = useLiveStream();
  const [history, setHistory] = useState<TimelinePoint[]>([]);
  const [config, setConfig] = useState<RunConfig | null>(null);
  const [activeRegion, setActiveRegion] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const activeTimer = useRef<number | null>(null);

  // Adopt the config from the handshake once.
  useEffect(() => {
    if (snapshot && !config) setConfig(snapshot.config);
  }, [snapshot, config]);

  // Append to the timeline, and clear it when the stream drops so a gap is
  // visible rather than hidden behind old data.
  useEffect(() => {
    if (!frame) return;
    if (frame.schema_version !== EXPECTED_SCHEMA_VERSION) {
      setActionError(
        `frame schema v${frame.schema_version} does not match expected v${EXPECTED_SCHEMA_VERSION}`,
      );
      return;
    }
    setHistory((prev) => {
      const next = [...prev, toTimelinePoint(frame.step, frame.spikes.length, frame.readouts)];
      return next.length > HISTORY_LIMIT ? next.slice(next.length - HISTORY_LIMIT) : next;
    });
  }, [frame]);

  useEffect(() => {
    if (state !== "open") setHistory([]);
  }, [state]);

  const guard = useCallback(async (fn: () => Promise<unknown>) => {
    try {
      setActionError(null);
      await fn();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    }
  }, []);

  const handleStart = useCallback(() => {
    void guard(() => api.start());
  }, [guard]);

  const handlePause = useCallback(() => {
    void guard(() => api.pause());
  }, [guard]);

  const handleReset = useCallback(() => {
    void guard(async () => {
      const res = await api.reset();
      setConfig(res.state.config);
      setHistory([]);
    });
  }, [guard]);

  const handleStimulate = useCallback(
    (region: string, strength: number, durationMs: number) => {
      setActiveRegion(region);
      if (activeTimer.current !== null) window.clearTimeout(activeTimer.current);
      activeTimer.current = window.setTimeout(() => setActiveRegion(null), durationMs);
      void guard(() => api.stimulate(region, strength, durationMs));
    },
    [guard],
  );

  const handleConfig = useCallback(
    (update: Partial<RunConfig>) => {
      setConfig((prev) => (prev ? { ...prev, ...update } : prev));
      void guard(async () => {
        const res = await api.setConfig(update);
        setConfig(res.config);
      });
    },
    [guard],
  );

  const regions = snapshot?.regions ?? [];
  const positions = snapshot?.positions ?? [];
  const readouts = frame?.readouts ?? [];
  const disclaimer =
    snapshot?.readout_disclaimer ??
    "Affective readouts are engineering proxies over simulation state. The model has no feelings.";

  const live = state === "open" && frame !== null;
  const running = frame?.status.running ?? false;

  const headerStats = useMemo(() => {
    if (!frame) return null;
    return {
      step: frame.step,
      t: frame.t_ms,
      spikes: frame.spikes.length,
    };
  }, [frame]);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" />
          <div>
            <h1>mem20-brainz</h1>
            <p>live simulation observatory</p>
          </div>
        </div>

        <div className="topbar-stats">
          {headerStats ? (
            <>
              <span>
                step <code>{headerStats.step.toLocaleString()}</code>
              </span>
              <span>
                t <code>{headerStats.t.toFixed(1)} ms</code>
              </span>
              <span>
                spikes <code>{headerStats.spikes}</code>
              </span>
            </>
          ) : (
            <span className="muted">no live data</span>
          )}
        </div>

        <div className="topbar-note">
          synthetic network · tier0 point-LIF · not a model of a brain
        </div>
      </header>

      {!live ? (
        <Disconnected state={state} error={error ?? actionError} onReconnect={reconnect} />
      ) : (
        <main className="layout">
          <section className="panel viz-panel">
            <header className="panel-head">
              <h2>population</h2>
              <span className="badge">{frame.status.n_neurons} neurons</span>
            </header>
            <NeuronCanvas
              frame={frame}
              regions={regions}
              positions={positions}
              activeRegion={activeRegion}
            />
            <div className="region-legend">
              {frame.regions.map((r) => (
                <div key={r.name} className="legend-item">
                  <span className="legend-swatch" data-region={r.name} />
                  <span className="legend-name">{r.name}</span>
                  <span className="legend-rate mono">{r.rate_hz.toFixed(1)} Hz</span>
                  <span className="legend-active mono">
                    {(r.active_fraction * 100).toFixed(0)}% active
                  </span>
                </div>
              ))}
            </div>
          </section>

          <section className="panel timeline-panel">
            <header className="panel-head">
              <h2>timeline</h2>
              <span className="badge">{history.length} frames</span>
            </header>
            <Timeline history={history} traceNames={TRACE_NAMES} />
            <p className="panel-foot">
              Top: spikes per frame. Bottom: readout traces over the same frames. Valence is
              signed and drawn about the midline.
            </p>
          </section>

          <ReadoutPanel readouts={readouts} disclaimer={disclaimer} />

          <Controls
            running={running}
            regions={regions}
            config={config}
            disabled={!live}
            onStart={handleStart}
            onPause={handlePause}
            onReset={handleReset}
            onStimulate={handleStimulate}
            onConfig={handleConfig}
          />
        </main>
      )}

      {actionError && live && <div className="toast">{actionError}</div>}

      <StatusBar
        state={state}
        status={frame?.status ?? snapshot?.status ?? null}
        provenance={frame?.provenance ?? snapshot?.provenance ?? null}
        frameCount={frameCount}
        error={error}
      />
    </div>
  );
}
