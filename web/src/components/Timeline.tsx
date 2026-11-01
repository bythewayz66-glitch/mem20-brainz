/**
 * Timeline strip: recent spike counts and the readout traces over time.
 *
 * Both series are drawn from the frames actually received, so the strip is a
 * record of the stream, not a decoration. When the socket drops the history is
 * cleared by the parent, so a gap is visible rather than papered over.
 */

import { useEffect, useRef } from "react";
import type { Readout } from "../lib/types";

export interface TimelinePoint {
  step: number;
  spikes: number;
  readouts: Record<string, number>;
}

interface Props {
  history: TimelinePoint[];
  /** Readout names to draw as traces. */
  traceNames: string[];
}

const TRACE_COLORS: Record<string, string> = {
  arousal: "#4cc9f0",
  valence: "#90be6d",
  surprise: "#ffd166",
  confusion: "#b5179e",
  fear_like: "#f72585",
};

export function Timeline({ history, traceNames }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth;
    const cssH = canvas.clientHeight;
    if (canvas.width !== Math.floor(cssW * dpr) || canvas.height !== Math.floor(cssH * dpr)) {
      canvas.width = Math.floor(cssW * dpr);
      canvas.height = Math.floor(cssH * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#08090d";
    ctx.fillRect(0, 0, cssW, cssH);

    if (history.length < 2) {
      ctx.fillStyle = "rgba(255,255,255,0.3)";
      ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.textAlign = "center";
      ctx.fillText("collecting…", cssW / 2, cssH / 2);
      return;
    }

    const spikeH = cssH * 0.34;
    const traceTop = spikeH + 8;
    const traceH = cssH - traceTop - 4;

    // --- spike histogram -------------------------------------------------
    const maxSpikes = Math.max(1, ...history.map((p) => p.spikes));
    const barW = cssW / history.length;
    for (let i = 0; i < history.length; i += 1) {
      const h = (history[i].spikes / maxSpikes) * spikeH;
      ctx.fillStyle = "rgba(76,201,240,0.55)";
      ctx.fillRect(i * barW, spikeH - h, Math.max(barW - 0.5, 0.6), h);
    }
    ctx.strokeStyle = "rgba(255,255,255,0.10)";
    ctx.beginPath();
    ctx.moveTo(0, spikeH + 0.5);
    ctx.lineTo(cssW, spikeH + 0.5);
    ctx.stroke();

    // --- readout traces --------------------------------------------------
    // valence is signed [-1,1]; everything else is [0,1]. Draw valence about
    // the midline of the trace band so its sign is visible.
    for (const name of traceNames) {
      const color = TRACE_COLORS[name] ?? "#8d99ae";
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < history.length; i += 1) {
        const value = history[i].readouts[name];
        if (value === undefined) continue;
        const x = i * barW + barW / 2;
        const norm = name === "valence" ? (value + 1) / 2 : value;
        const y = traceTop + traceH - norm * traceH;
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }

    // Legend.
    ctx.font = "9px ui-monospace, SFMono-Regular, Menlo, monospace";
    ctx.textAlign = "left";
    let lx = 6;
    for (const name of traceNames) {
      ctx.fillStyle = TRACE_COLORS[name] ?? "#8d99ae";
      ctx.fillRect(lx, 4, 6, 6);
      ctx.fillStyle = "rgba(255,255,255,0.55)";
      ctx.fillText(name, lx + 9, 10);
      lx += ctx.measureText(name).width + 22;
    }
  }, [history, traceNames]);

  return (
    <div className="timeline-wrap">
      <canvas ref={canvasRef} className="timeline-canvas" />
    </div>
  );
}

export function toTimelinePoint(
  step: number,
  spikes: number,
  readouts: Readout[],
): TimelinePoint {
  const values: Record<string, number> = {};
  for (const r of readouts) values[r.name] = r.value;
  return { step, spikes, readouts: values };
}
