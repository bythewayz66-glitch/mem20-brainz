"""Tests for the live observability server: endpoints, frame schema, streaming.

These use FastAPI's TestClient, so they exercise the real app object, the real
runner and the real engine — no mocks. The WebSocket test drives an actual
stream and asserts on the frames that come back.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mem20_brainz.server import app as app_module
from mem20_brainz.server.frames import FRAME_SCHEMA_VERSION, bin_v, quantise_v
from mem20_brainz.server.runner import RunConfig, RunRunner


@pytest.fixture()
def client() -> TestClient:
    """A client over a fresh runner, so tests do not share simulation state."""
    app_module.set_runner(RunRunner(RunConfig(steps_per_frame=2)))
    with TestClient(app_module.app) as c:
        yield c
    app_module.set_runner(None)


class TestHealthAndState:
    def test_health(self, client: TestClient) -> None:
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["n_neurons"] > 0
        assert body["n_synapses"] > 0
        assert body["data_kind"] == "synthetic"

    def test_state_snapshot(self, client: TestClient) -> None:
        r = client.get("/api/state")
        assert r.status_code == 200
        body = r.json()
        assert body["schema_version"] == 1
        assert len(body["regions"]) == 5
        assert len(body["positions"]) == body["status"]["n_neurons"]
        assert body["readout_provenance"] == "derived_proxy"
        assert "no feelings" in body["readout_disclaimer"]

    def test_state_declares_synthetic_provenance(self, client: TestClient) -> None:
        prov = client.get("/api/state").json()["provenance"]
        assert prov["kind"] == "synthetic"
        assert prov["generator"]
        assert prov["seed"] is not None
        assert prov["fitted_to"] is None


class TestRunControl:
    def test_start_and_pause(self, client: TestClient) -> None:
        assert client.post("/api/run/start").json()["running"] is True
        assert client.get("/api/health").json()["running"] is True
        assert client.post("/api/run/pause").json()["running"] is False
        assert client.get("/api/health").json()["running"] is False

    def test_reset_restores_initial_state(self, client: TestClient) -> None:
        client.post("/api/run/start")
        for _ in range(5):
            client.get("/api/frame")
        assert client.get("/api/health").json()["running"] is True
        body = client.post("/api/run/reset").json()
        assert body["ok"] is True
        # Reset rewinds to the warmed-up start, not to step 0: the runner
        # advances WARMUP_STEPS before serving so the network is already firing.
        from mem20_brainz.server.runner import WARMUP_STEPS

        assert body["state"]["status"]["steps"] == WARMUP_STEPS
        assert body["state"]["status"]["running"] is False

    def test_reset_rewinds_progress(self, client: TestClient) -> None:
        """Reset must discard accumulated steps, not merely pause."""
        client.post("/api/run/start")
        for _ in range(10):
            client.get("/api/frame")
        before = client.get("/api/state").json()["status"]["steps"]
        after = client.post("/api/run/reset").json()["state"]["status"]
        assert after["steps"] < before
        assert after["frames_emitted"] == 0

    def test_warm_up_leaves_network_firing(self, client: TestClient) -> None:
        """A freshly built runner must already be active, not silent."""
        frame = client.get("/api/frame").json()
        total = sum(r["rate_hz"] for r in frame["regions"])
        assert total > 0.0, "network is silent immediately after warm-up"

    def test_step_advances_exactly_one_frame(self, client: TestClient) -> None:
        first = client.post("/api/run/step").json()
        second = client.post("/api/run/step").json()
        assert second["step"] > first["step"]


class TestConfig:
    def test_update_dt(self, client: TestClient) -> None:
        r = client.post("/api/config", json={"dt_ms": 0.05})
        assert r.status_code == 200
        assert r.json()["config"]["dt_ms"] == pytest.approx(0.05)

    def test_update_baseline(self, client: TestClient) -> None:
        r = client.post("/api/config", json={"baseline_rate_hz": 20.0})
        assert r.json()["config"]["baseline_rate_hz"] == pytest.approx(20.0)

    def test_rejects_unknown_key(self, client: TestClient) -> None:
        # Pydantic drops unknown fields, so this must not 500.
        r = client.post("/api/config", json={"not_a_key": 1})
        assert r.status_code == 200

    def test_rejects_invalid_dt(self, client: TestClient) -> None:
        assert client.post("/api/config", json={"dt_ms": -1.0}).status_code == 422


class TestStimulus:
    def test_inject_into_known_region(self, client: TestClient) -> None:
        r = client.post(
            "/api/stimulus", json={"region": "assoc_a", "strength": 8.0, "duration_ms": 100.0}
        )
        assert r.status_code == 200
        stim = r.json()["stimulus"]
        assert stim["region"] == "assoc_a"
        assert stim["strength"] == pytest.approx(8.0)
        assert stim["remaining_steps"] > 0

    def test_unknown_region_is_404(self, client: TestClient) -> None:
        r = client.post("/api/stimulus", json={"region": "nope", "strength": 1.0})
        assert r.status_code == 404

    def test_zero_duration_is_422(self, client: TestClient) -> None:
        r = client.post("/api/stimulus", json={"region": "assoc_a", "duration_ms": 0.0})
        assert r.status_code == 422

    def test_stimulus_raises_activity(self, client: TestClient) -> None:
        """A strong stimulus must visibly move the measured activity."""
        for _ in range(60):
            client.get("/api/frame")
        before = client.get("/api/frame").json()
        before_threat = next(r for r in before["regions"] if r["name"] == "threat")["rate_hz"]

        client.post(
            "/api/stimulus", json={"region": "threat", "strength": 12.0, "duration_ms": 500.0}
        )
        for _ in range(40):
            after = client.get("/api/frame").json()
        after_threat = next(r for r in after["regions"] if r["name"] == "threat")["rate_hz"]

        assert after_threat > before_threat, (
            f"stimulating threat did not raise its rate: {before_threat} -> {after_threat}"
        )


class TestFrameSchema:
    def test_frame_has_every_required_field(self, client: TestClient) -> None:
        frame = client.get("/api/frame").json()
        assert set(frame) == {
            "schema_version",
            "step",
            "t_ms",
            "dt_ms",
            "v_q",
            "v_range",
            "v_bin_size",
            "spikes",
            "regions",
            "readouts",
            "status",
            "provenance",
        }
        assert frame["schema_version"] == FRAME_SCHEMA_VERSION

    def test_quantised_voltages_are_in_range(self, client: TestClient) -> None:
        frame = client.get("/api/frame").json()
        assert all(0 <= q <= 255 for q in frame["v_q"])
        assert len(frame["v_range"]) == 2
        assert frame["v_range"][0] < frame["v_range"][1]

    def test_spikes_are_valid_neuron_indices(self, client: TestClient) -> None:
        frame = client.get("/api/frame").json()
        n = frame["status"]["n_neurons"]
        assert all(0 <= s < n for s in frame["spikes"])

    def test_region_aggregates_cover_every_region(self, client: TestClient) -> None:
        frame = client.get("/api/frame").json()
        names = [r["name"] for r in frame["regions"]]
        assert names == ["sensory", "assoc_a", "assoc_b", "threat", "inhib"]
        for region in frame["regions"]:
            assert region["rate_hz"] >= 0.0
            assert 0.0 <= region["active_fraction"] <= 1.0
            assert region["n_neurons"] > 0

    def test_readouts_carry_provenance_and_inputs(self, client: TestClient) -> None:
        frame = client.get("/api/frame").json()
        assert len(frame["readouts"]) == 5
        for readout in frame["readouts"]:
            assert readout["provenance"] == "derived_proxy"
            assert readout["inputs"], f"{readout['name']} has no inputs"
            assert readout["formula"]
            assert 0.0 <= readout["value"] <= 1.0 or readout["name"] == "valence"

    def test_status_reports_backend_and_scale(self, client: TestClient) -> None:
        status = client.get("/api/frame").json()["status"]
        assert status["backend"]
        assert status["n_neurons"] > 0
        assert status["n_synapses"] > 0
        assert status["data_kind"] == "synthetic"
        assert "steps_per_sec" in status

    def test_frame_provenance_is_synthetic(self, client: TestClient) -> None:
        prov = client.get("/api/frame").json()["provenance"]
        assert prov["kind"] == "synthetic"
        assert prov["parameters_are_placeholders"] is True


class TestWebSocket:
    def test_hello_then_frames(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello"
            assert hello["state"]["readout_provenance"] == "derived_proxy"

            client.post("/api/run/start")
            frame_msg = ws.receive_json()
            assert frame_msg["type"] == "frame"
            frame = frame_msg["frame"]
            assert frame["schema_version"] == FRAME_SCHEMA_VERSION
            assert "readouts" in frame
            assert len(frame["readouts"]) == 5

    def test_ping_pong(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()  # hello
            ws.send_json({"type": "ping"})
            # The pump may emit a frame first if the run is started; it is not,
            # so the next message must be the pong.
            msg = ws.receive_json()
            assert msg["type"] == "pong"

    def test_no_frames_while_paused(self, client: TestClient) -> None:
        """A paused run must not stream: the client should see only the hello."""
        with client.websocket_connect("/ws") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello"
            ws.send_json({"type": "ping"})
            msg = ws.receive_json()
            assert msg["type"] == "pong", "a paused run streamed a frame"


class TestFrameHelpers:
    def test_quantise_maps_range_endpoints(self) -> None:
        import numpy as np

        q = quantise_v(np.array([-80.0, -60.0, -40.0]), -80.0, -40.0)
        assert q[0] == 0
        assert q[-1] == 255
        assert 0 < q[1] < 255

    def test_quantise_clips_out_of_range(self) -> None:
        import numpy as np

        q = quantise_v(np.array([-200.0, 200.0]), -80.0, -40.0)
        assert q == [0, 255]

    def test_bin_v_reduces_length(self) -> None:
        import numpy as np

        v = np.arange(1000, dtype=np.float64)
        binned, size = bin_v(v, 100)
        assert size == 10
        assert len(binned) == 100

    def test_bin_v_noop_when_small(self) -> None:
        import numpy as np

        v = np.arange(50, dtype=np.float64)
        binned, size = bin_v(v, 100)
        assert size == 1
        assert len(binned) == 50


class TestHeadless:
    def test_headless_run_produces_trace(self) -> None:
        from mem20_brainz.server.headless import run_headless

        result = run_headless(steps=30, print_every=10, config=RunConfig(steps_per_frame=2))
        assert result["n_neurons"] > 0
        assert result["data_kind"] == "synthetic"
        assert len(result["trace"]) >= 3
        assert result["final_frame"]["schema_version"] == FRAME_SCHEMA_VERSION

    def test_headless_stimulus_changes_activity(self) -> None:
        from mem20_brainz.server.headless import run_headless

        result = run_headless(
            steps=120,
            stimulus_region="threat",
            stimulus_strength=12.0,
            stimulus_at=20,
            stimulus_duration_ms=500.0,
            print_every=20,
            config=RunConfig(steps_per_frame=2),
        )
        rates = [row["readouts"]["arousal"] for row in result["trace"]]
        assert max(rates) > min(rates), "arousal never moved across the run"
