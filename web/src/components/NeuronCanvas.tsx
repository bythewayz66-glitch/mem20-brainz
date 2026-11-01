/**
 * Canvas rendering of the neuron population.
 *
 * Draws one dot per neuron at its display position, coloured by region, with
 * brightness driven by the quantised membrane potential. Spiking neurons get a
 * bright ring so a spike is visible even when the voltage is mid-range.
 *
 * The positions are display coordinates from the scenario, not soma positions
 * in any anatomical reference space. The caption says so.
 */

import { useEffect, useRef } from "react";
import type { Frame, RegionInfo } from "../lib/types";
import { regionColor } from "../lib/types";

interface Props {
  frame: Frame | null;
  regions: RegionInfo[];
  positions: number[][];
  /** Region currently being stimulated, highlighted in the canvas. */
  activeRegion: string | null;
}

const PADDING = 18;

export function NeuronCanvas({ frame, regions, positions, activeRegion }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const spikesRef = useRef<Set<number>>(new Set());
  const spikeAgeRef = useRef<Map<number, number>>(new Map());

  // Track recent spikes so the ring fades over a few frames instead of
  // flashing for a single 16 ms paint.
  useEffect(() => {
    if (!frame) return;
    const now = performance.now();
    for (const id of frame.spikes) spikeAgeRef.current.set(id, now);
    spikesRef.current = new Set(frame.spikes);
  }, [frame]);

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

    const size = Math.min(cssW, cssH) - PADDING * 2;
    const ox = (cssW - size) / 2;
    const oy = (cssH - size) / 2;

    // Region halos, so the blocks are legible even when quiet.
    for (const region of regions) {
      const [cx, cy] = region.center;
      const isActive = region.name === activeRegion;
      ctx.beginPath();
      ctx.arc(ox + cx * size, oy + cy * size, size * 0.135, 0, Math.PI * 2);
      ctx.fillStyle = isActive ? "rgba(247,37,133,0.10)" : "rgba(255,255,255,0.022)";
      ctx.fill();
      ctx.strokeStyle = isActive ? "rgba(247,37,133,0.55)" : "rgba(255,255,255,0.07)";
      ctx.lineWidth = isActive ? 1.5 : 1;
      ctx.stroke();

      ctx.fillStyle = isActive ? "#f72585" : "rgba(255,255,255,0.42)";
      ctx.font = "600 10px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.textAlign = "center";
      ctx.fillText(region.name, ox + cx * size, oy + cy * size - size * 0.135 - 6);
    }

    if (!frame) {
      ctx.fillStyle = "rgba(255,255,255,0.35)";
      ctx.font = "13px ui-monospace, SFMono-Regular, Menlo, monospace";
      ctx.textAlign = "center";
      ctx.fillText("no live data", cssW / 2, cssH / 2);
      return;
    }

    const [vMin, vMax] = frame.v_range;
    const binSize = frame.v_bin_size;
    const now = performance.now();
    const spikeSet = spikesRef.current;
    const ages = spikeAgeRef.current;

    // Map each neuron to its region colour once per paint.
    const colorFor = (i: number): string => {
      for (const region of regions) {
        if (i < region.n_neurons) return regionColor(region.name);
      }
      return "#8d99ae";
    };
    // Precompute region boundaries for O(regions) lookup instead of O(n).
    const bounds: { end: number; color: string }[] = [];
    let acc = 0;
    for (const region of regions) {
      acc += region.n_neurons;
      bounds.push({ end: acc, color: regionColor(region.name) });
    }
    const colorAt = (i: number): string => {
      for (const b of bounds) if (i < b.end) return b.color;
      return "#8d99ae";
    };
    void colorFor;

    const n = positions.length;
    for (let i = 0; i < n; i += 1) {
      const pos = positions[i];
      if (!pos) continue;
      const x = ox + pos[0] * size;
      const y = oy + pos[1] * size;

      // Voltage for this neuron: binned frames average groups of neurons.
      const qIndex = Math.min(Math.floor(i / binSize), frame.v_q.length - 1);
      const q = frame.v_q[qIndex] ?? 0;
      const norm = q / 255;
      const v = vMin + norm * (vMax - vMin);

      const color = colorAt(i);
      const alpha = 0.18 + 0.82 * norm;
      ctx.globalAlpha = alpha;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, y, 2.1, 0, Math.PI * 2);
      ctx.fill();

      // Spike ring, fading over ~180 ms.
      const age = ages.get(i);
      if (age !== undefined) {
        const dt = now - age;
        if (dt < 180) {
          const fade = 1 - dt / 180;
          ctx.globalAlpha = fade;
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          ctx.arc(x, y, 2.1 + 3.2 * fade, 0, Math.PI * 2);
          ctx.stroke();
        } else {
          ages.delete(i);
        }
      }
      void v;
    }
    ctx.globalAlpha = 1;
    void spikeSet;
  }, [frame, regions, positions, activeRegion]);

  return (
    <div className="canvas-wrap">
      <canvas ref={canvasRef} className="neuron-canvas" />
      <div className="canvas-caption">
        display coordinates only — not soma positions in any anatomical space
      </div>
    </div>
  );
}
