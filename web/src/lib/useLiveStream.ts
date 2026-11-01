/**
 * WebSocket client for the live simulation stream.
 *
 * Honesty requirement: when the socket drops, this hook reports `closed` and
 * clears the live frame. The UI must then show a disconnected state rather than
 * freezing on the last frame it saw — a stale frame presented as live would be
 * a lie about the state of the simulation.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { ConnectionState, Frame, Snapshot } from "./types";

const RECONNECT_DELAY_MS = 1500;

function wsUrl(): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws`;
}

export interface LiveStream {
  state: ConnectionState;
  snapshot: Snapshot | null;
  frame: Frame | null;
  /** Frames received since mount. Used to prove the stream is live. */
  frameCount: number;
  error: string | null;
  reconnect: () => void;
}

export function useLiveStream(): LiveStream {
  const [state, setState] = useState<ConnectionState>("connecting");
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [frame, setFrame] = useState<Frame | null>(null);
  const [frameCount, setFrameCount] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const socketRef = useRef<WebSocket | null>(null);
  const retryRef = useRef<number | null>(null);
  const closedByUs = useRef(false);

  const connect = useCallback(() => {
    if (socketRef.current) {
      closedByUs.current = true;
      socketRef.current.close();
      socketRef.current = null;
    }
    closedByUs.current = false;
    setState("connecting");
    setError(null);

    let socket: WebSocket;
    try {
      socket = new WebSocket(wsUrl());
    } catch (err) {
      setState("error");
      setError(err instanceof Error ? err.message : "could not open socket");
      return;
    }
    socketRef.current = socket;

    socket.onopen = () => {
      setState("open");
      setError(null);
    };

    socket.onmessage = (event) => {
      let msg: { type?: string; state?: Snapshot; frame?: Frame };
      try {
        msg = JSON.parse(event.data as string);
      } catch {
        return;
      }
      if (msg.type === "hello" && msg.state) {
        setSnapshot(msg.state);
      } else if (msg.type === "frame" && msg.frame) {
        setFrame(msg.frame);
        setFrameCount((n) => n + 1);
      }
    };

    socket.onerror = () => {
      setState("error");
      setError("socket error");
    };

    socket.onclose = () => {
      socketRef.current = null;
      if (closedByUs.current) return;
      // Drop the frame: a frozen last frame must not be shown as live.
      setFrame(null);
      setState("closed");
      retryRef.current = window.setTimeout(connect, RECONNECT_DELAY_MS);
    };
  }, []);

  useEffect(() => {
    connect();
    return () => {
      closedByUs.current = true;
      if (retryRef.current !== null) window.clearTimeout(retryRef.current);
      socketRef.current?.close();
      socketRef.current = null;
    };
  }, [connect]);

  return { state, snapshot, frame, frameCount, error, reconnect: connect };
}
