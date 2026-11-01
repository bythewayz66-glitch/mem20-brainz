/**
 * Readout panel: one gauge per named state, with the arithmetic exposed.
 *
 * Every gauge shows the value, the raw inputs it was computed from, and the
 * formula. The "derived proxy — not a feeling" label is persistent and is not
 * dismissible: it is part of the panel, not a tooltip.
 */

import { useState } from "react";
import type { Readout } from "../lib/types";

interface Props {
  readouts: Readout[];
  disclaimer: string;
}

const ACCENT: Record<string, string> = {
  arousal: "#4cc9f0",
  valence: "#90be6d",
  surprise: "#ffd166",
  confusion: "#b5179e",
  fear_like: "#f72585",
};

function formatInput(value: number): string {
  if (!Number.isFinite(value)) return String(value);
  if (Math.abs(value) >= 1000) return value.toFixed(0);
  if (Math.abs(value) >= 10) return value.toFixed(2);
  return value.toFixed(4);
}

function Gauge({ readout }: { readout: Readout }) {
  const [open, setOpen] = useState(false);
  const accent = ACCENT[readout.name] ?? "#8d99ae";
  const signed = readout.name === "valence";
  // valence is [-1,1]; everything else is [0,1].
  const pct = signed ? ((readout.value + 1) / 2) * 100 : readout.value * 100;

  return (
    <div className={`gauge ${open ? "gauge-open" : ""}`}>
      <button
        type="button"
        className="gauge-head"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        title="Show the inputs and formula behind this number"
      >
        <span className="gauge-name">{readout.name.replace(/_/g, " ")}</span>
        <span className="gauge-value" style={{ color: accent }}>
          {readout.value.toFixed(3)}
        </span>
        <span className="gauge-caret">{open ? "−" : "+"}</span>
      </button>

      <div className="gauge-track">
        {signed && <div className="gauge-mid" />}
        <div
          className="gauge-fill"
          style={{
            background: accent,
            width: signed ? `${Math.abs(readout.value) * 50}%` : `${pct}%`,
            left: signed ? (readout.value >= 0 ? "50%" : `${50 - Math.abs(readout.value) * 50}%`) : "0",
          }}
        />
      </div>

      <div className="gauge-proxy">derived proxy — not a feeling</div>

      {open && (
        <div className="gauge-detail">
          <p className="gauge-interp">{readout.interpretation}</p>
          <div className="gauge-formula">
            <span className="label">formula</span>
            <code>{readout.formula}</code>
          </div>
          <div className="gauge-inputs">
            <span className="label">measured inputs</span>
            <table>
              <tbody>
                {Object.entries(readout.inputs).map(([key, value]) => (
                  <tr key={key}>
                    <td className="input-key">{key}</td>
                    <td className="input-value">{formatInput(value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="gauge-prov">
            provenance: <code>{readout.provenance}</code>
          </div>
        </div>
      )}
    </div>
  );
}

export function ReadoutPanel({ readouts, disclaimer }: Props) {
  return (
    <section className="panel readout-panel">
      <header className="panel-head">
        <h2>affective readout</h2>
        <span className="badge badge-warn">derived proxy</span>
      </header>

      <p className="disclaimer">{disclaimer}</p>

      <div className="gauges">
        {readouts.map((r) => (
          <Gauge key={r.name} readout={r} />
        ))}
      </div>

      <p className="panel-foot">
        Expand any gauge to see the measured quantities and the exact formula. These are
        engineering proxies over simulation state; the model has no feelings.
      </p>
    </section>
  );
}
