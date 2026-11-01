/** Controls: play/pause, reset, stimulus injection, parameter sliders. */

import { useState } from "react";
import type { RegionInfo, RunConfig } from "../lib/types";

interface Props {
  running: boolean;
  regions: RegionInfo[];
  config: RunConfig | null;
  disabled: boolean;
  onStart: () => void;
  onPause: () => void;
  onReset: () => void;
  onStimulate: (region: string, strength: number, durationMs: number) => void;
  onConfig: (update: Partial<RunConfig>) => void;
}

export function Controls({
  running,
  regions,
  config,
  disabled,
  onStart,
  onPause,
  onReset,
  onStimulate,
  onConfig,
}: Props) {
  const [region, setRegion] = useState<string>("assoc_a");
  const [strength, setStrength] = useState(10);
  const [duration, setDuration] = useState(300);

  const target = regions.find((r) => r.name === region) ?? regions[0];

  return (
    <section className="panel controls-panel">
      <header className="panel-head">
        <h2>controls</h2>
      </header>

      <div className="button-row">
        <button
          type="button"
          className={running ? "btn btn-ghost" : "btn btn-primary"}
          onClick={running ? onPause : onStart}
          disabled={disabled}
        >
          {running ? "pause" : "play"}
        </button>
        <button type="button" className="btn btn-ghost" onClick={onReset} disabled={disabled}>
          reset
        </button>
      </div>

      <div className="control-group">
        <label className="control-label" htmlFor="stim-region">
          stimulus target
        </label>
        <select
          id="stim-region"
          value={region}
          onChange={(e) => setRegion(e.target.value)}
          disabled={disabled}
        >
          {regions.map((r) => (
            <option key={r.name} value={r.name}>
              {r.name} ({r.n_neurons} neurons, {r.kind})
            </option>
          ))}
        </select>
      </div>

      <div className="control-group">
        <label className="control-label" htmlFor="stim-strength">
          strength <span className="mono">{strength.toFixed(1)}</span>
        </label>
        <input
          id="stim-strength"
          type="range"
          min={-20}
          max={30}
          step={0.5}
          value={strength}
          onChange={(e) => setStrength(Number(e.target.value))}
          disabled={disabled}
        />
      </div>

      <div className="control-group">
        <label className="control-label" htmlFor="stim-duration">
          duration <span className="mono">{duration} ms</span>
        </label>
        <input
          id="stim-duration"
          type="range"
          min={20}
          max={2000}
          step={20}
          value={duration}
          onChange={(e) => setDuration(Number(e.target.value))}
          disabled={disabled}
        />
      </div>

      <button
        type="button"
        className="btn btn-accent"
        onClick={() => onStimulate(region, strength, duration)}
        disabled={disabled || !target}
      >
        inject stimulus
      </button>

      <hr className="rule" />

      <div className="control-group">
        <label className="control-label" htmlFor="steps-per-frame">
          steps / frame <span className="mono">{config?.steps_per_frame ?? "—"}</span>
        </label>
        <input
          id="steps-per-frame"
          type="range"
          min={1}
          max={20}
          step={1}
          value={config?.steps_per_frame ?? 4}
          onChange={(e) => onConfig({ steps_per_frame: Number(e.target.value) })}
          disabled={disabled}
        />
      </div>

      <div className="control-group">
        <label className="control-label" htmlFor="baseline">
          baseline rate <span className="mono">{config?.baseline_rate_hz ?? "—"} Hz</span>
        </label>
        <input
          id="baseline"
          type="range"
          min={1}
          max={40}
          step={0.5}
          value={config?.baseline_rate_hz ?? 12}
          onChange={(e) => onConfig({ baseline_rate_hz: Number(e.target.value) })}
          disabled={disabled}
        />
        <p className="hint">
          The reference the arousal gauge divides by. Changing it rescales arousal, not the
          network.
        </p>
      </div>

      <div className="control-group">
        <label className="control-label" htmlFor="tonic">
          tonic drive <span className="mono">{config?.tonic_current ?? "—"}</span>
        </label>
        <input
          id="tonic"
          type="range"
          min={0}
          max={6}
          step={0.1}
          value={config?.tonic_current ?? 2}
          onChange={(e) => onConfig({ tonic_current: Number(e.target.value) })}
          disabled={disabled}
        />
        <p className="hint">Below ~1.9 the network is silent and every gauge reads zero.</p>
      </div>
    </section>
  );
}
