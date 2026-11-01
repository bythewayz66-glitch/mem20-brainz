"""Unit tests for the affective readout proxies.

Each test pins a known input to an expected output, so the arithmetic is
verifiable by hand. The point of these tests is not that the numbers are
*meaningful* — they are proxies — but that they are *correct* and that every
readout carries its inputs, formula and provenance tag.
"""

from __future__ import annotations

import pytest

from mem20_brainz.readout import (
    DERIVED_PROXY,
    ReadoutInputs,
    arousal,
    compute_all,
    confusion,
    fear_like,
    surprise,
    synchrony_index,
    to_dicts,
    valence,
)


class TestArousal:
    def test_at_baseline_is_half_scale(self) -> None:
        # ratio = 1.0, value = 1.0 / 2 = 0.5
        r = arousal(ReadoutInputs(rate_hz=10.0, baseline_rate_hz=10.0))
        assert r.value == pytest.approx(0.5)

    def test_double_baseline_saturates(self) -> None:
        r = arousal(ReadoutInputs(rate_hz=20.0, baseline_rate_hz=10.0))
        assert r.value == pytest.approx(1.0)

    def test_zero_rate_is_zero(self) -> None:
        r = arousal(ReadoutInputs(rate_hz=0.0, baseline_rate_hz=10.0))
        assert r.value == pytest.approx(0.0)

    def test_above_double_baseline_clips(self) -> None:
        r = arousal(ReadoutInputs(rate_hz=1000.0, baseline_rate_hz=10.0))
        assert r.value == pytest.approx(1.0)

    def test_zero_baseline_does_not_divide_by_zero(self) -> None:
        r = arousal(ReadoutInputs(rate_hz=5.0, baseline_rate_hz=0.0))
        assert r.value == pytest.approx(1.0)

    def test_reports_inputs_and_formula(self) -> None:
        r = arousal(ReadoutInputs(rate_hz=10.0, baseline_rate_hz=10.0))
        assert r.inputs["rate_hz"] == 10.0
        assert r.inputs["baseline_rate_hz"] == 10.0
        assert r.inputs["ratio"] == pytest.approx(1.0)
        assert "baseline_rate_hz" in r.formula
        assert r.provenance == DERIVED_PROXY


class TestValence:
    def test_pure_excitation_is_plus_one(self) -> None:
        r = valence(ReadoutInputs(exc_drive=10.0, inh_drive=0.0))
        assert r.value == pytest.approx(1.0)

    def test_pure_inhibition_is_minus_one(self) -> None:
        r = valence(ReadoutInputs(exc_drive=0.0, inh_drive=10.0))
        assert r.value == pytest.approx(-1.0)

    def test_balanced_drive_is_zero(self) -> None:
        r = valence(ReadoutInputs(exc_drive=7.0, inh_drive=7.0))
        assert r.value == pytest.approx(0.0)

    def test_asymmetric_drive(self) -> None:
        # (30 - 10) / 40 = 0.5
        r = valence(ReadoutInputs(exc_drive=30.0, inh_drive=10.0))
        assert r.value == pytest.approx(0.5)

    def test_no_drive_is_zero_not_nan(self) -> None:
        r = valence(ReadoutInputs(exc_drive=0.0, inh_drive=0.0))
        assert r.value == pytest.approx(0.0)

    def test_is_signed(self) -> None:
        assert valence(ReadoutInputs(exc_drive=1.0, inh_drive=9.0)).value < 0.0


class TestSurprise:
    def test_at_expectation_is_zero(self) -> None:
        r = surprise(ReadoutInputs(rate_hz=10.0, expected_rate_hz=10.0, baseline_rate_hz=10.0))
        assert r.value == pytest.approx(0.0)

    def test_deviation_is_positive(self) -> None:
        # |20 - 10| / (10 + 10) = 0.5
        r = surprise(ReadoutInputs(rate_hz=20.0, expected_rate_hz=10.0, baseline_rate_hz=10.0))
        assert r.value == pytest.approx(0.5)

    def test_is_symmetric_in_sign(self) -> None:
        up = surprise(ReadoutInputs(rate_hz=20.0, expected_rate_hz=10.0, baseline_rate_hz=10.0))
        down = surprise(ReadoutInputs(rate_hz=0.0, expected_rate_hz=10.0, baseline_rate_hz=10.0))
        assert up.value == pytest.approx(down.value)

    def test_large_deviation_clips(self) -> None:
        r = surprise(ReadoutInputs(rate_hz=1000.0, expected_rate_hz=0.0, baseline_rate_hz=1.0))
        assert r.value == pytest.approx(1.0)

    def test_reports_absolute_deviation(self) -> None:
        r = surprise(ReadoutInputs(rate_hz=20.0, expected_rate_hz=10.0, baseline_rate_hz=10.0))
        assert r.inputs["abs_deviation_hz"] == pytest.approx(10.0)


class TestConfusion:
    def test_uniform_regions_is_zero(self) -> None:
        r = confusion(ReadoutInputs(region_rates_hz=[10.0, 10.0, 10.0]))
        assert r.value == pytest.approx(0.0)

    def test_single_region_is_zero(self) -> None:
        # CV is undefined for one sample; we report 0 rather than NaN.
        r = confusion(ReadoutInputs(region_rates_hz=[10.0]))
        assert r.value == pytest.approx(0.0)

    def test_no_regions_is_zero(self) -> None:
        r = confusion(ReadoutInputs(region_rates_hz=[]))
        assert r.value == pytest.approx(0.0)

    def test_dispersion_is_positive(self) -> None:
        # mean 10, population std of [0, 20] = 10, CV = 1.0
        r = confusion(ReadoutInputs(region_rates_hz=[0.0, 20.0]))
        assert r.value == pytest.approx(1.0)

    def test_partial_dispersion(self) -> None:
        # [5, 15]: mean 10, std 5, CV 0.5
        r = confusion(ReadoutInputs(region_rates_hz=[5.0, 15.0]))
        assert r.value == pytest.approx(0.5)

    def test_all_zero_regions_is_zero_not_nan(self) -> None:
        r = confusion(ReadoutInputs(region_rates_hz=[0.0, 0.0, 0.0]))
        assert r.value == pytest.approx(0.0)


class TestFearLike:
    def test_requires_both_factors(self) -> None:
        # High arousal alone must not produce a score.
        r = fear_like(ReadoutInputs(sustained_arousal=1.0, threat_synchrony=0.0))
        assert r.value == pytest.approx(0.0)
        # High synchrony alone must not either.
        r = fear_like(ReadoutInputs(sustained_arousal=0.0, threat_synchrony=1.0))
        assert r.value == pytest.approx(0.0)

    def test_product_of_factors(self) -> None:
        r = fear_like(ReadoutInputs(sustained_arousal=0.5, threat_synchrony=0.5))
        assert r.value == pytest.approx(0.25)

    def test_both_high_saturates(self) -> None:
        r = fear_like(ReadoutInputs(sustained_arousal=1.0, threat_synchrony=1.0))
        assert r.value == pytest.approx(1.0)

    def test_reports_both_inputs(self) -> None:
        r = fear_like(ReadoutInputs(sustained_arousal=0.4, threat_synchrony=0.6))
        assert r.inputs["sustained_arousal"] == pytest.approx(0.4)
        assert r.inputs["threat_synchrony"] == pytest.approx(0.6)


class TestSynchronyIndex:
    def test_constant_count_is_asynchronous(self) -> None:
        # Fano = 0 for a constant series, which is below 1 -> clipped to 0.
        assert synchrony_index([5, 5, 5, 5], 100) == pytest.approx(0.0)

    def test_poisson_like_is_near_zero(self) -> None:
        # mean == var -> Fano 1 -> 0
        assert synchrony_index([4, 4, 4, 4], 100) == pytest.approx(0.0)

    def test_bursty_is_positive(self) -> None:
        # mean 5, var 25 -> Fano 5 -> (5-1)/1.0 clipped to 1
        assert synchrony_index([0, 0, 0, 20], 100) == pytest.approx(1.0)

    def test_short_window_is_zero(self) -> None:
        assert synchrony_index([5], 100) == pytest.approx(0.0)
        assert synchrony_index([], 100) == pytest.approx(0.0)

    def test_silent_window_is_zero(self) -> None:
        assert synchrony_index([0, 0, 0], 100) == pytest.approx(0.0)

    def test_zero_neurons_is_zero(self) -> None:
        assert synchrony_index([1, 2, 3], 0) == pytest.approx(0.0)

    def test_bounded(self) -> None:
        for counts in ([0, 100], [1, 1, 1], [0, 0, 0, 0, 50], [3, 3, 3, 3]):
            value = synchrony_index(counts, 100)
            assert 0.0 <= value <= 1.0


class TestComputeAll:
    def test_returns_every_readout_in_order(self) -> None:
        readouts = compute_all(ReadoutInputs())
        assert [r.name for r in readouts] == [
            "arousal",
            "valence",
            "surprise",
            "confusion",
            "fear_like",
        ]

    def test_every_readout_is_tagged_as_a_proxy(self) -> None:
        for r in compute_all(ReadoutInputs()):
            assert r.provenance == DERIVED_PROXY

    def test_every_readout_carries_inputs_and_formula(self) -> None:
        for r in compute_all(ReadoutInputs()):
            assert r.inputs, f"{r.name} has no inputs"
            assert r.formula, f"{r.name} has no formula"
            assert r.interpretation, f"{r.name} has no interpretation"

    def test_values_are_finite(self) -> None:
        for r in compute_all(ReadoutInputs()):
            assert r.value == r.value  # not NaN
            assert abs(r.value) != float("inf")

    def test_serialises_to_dicts(self) -> None:
        payload = to_dicts(compute_all(ReadoutInputs()))
        assert len(payload) == 5
        for item in payload:
            assert set(item) == {
                "name",
                "value",
                "unit",
                "inputs",
                "formula",
                "provenance",
                "interpretation",
            }
            assert item["provenance"] == DERIVED_PROXY
