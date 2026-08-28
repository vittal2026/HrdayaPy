"""
inspect_pipeline/functions/beat_utils.py
===========================================
Shared beat-detection / average-beat utilities, backing Step 14c
(pick_and_compare_unipolar) and Step 14d (pick_and_compare_bipolar).

Ported from the experimental unipolar_representative_beat.py /
bipolar_representative_beat.py scripts' beat_utils.py, which was itself
validated against the Bratislava/EDGAR dataset. Generalised so it isn't
tied to one specific sim/gt variable pair, and independent of
pick_and_compare_multi.py's own (simpler, non-NaN-padded) inline
beat-detection helpers -- this module is the one used by the two
representative-beat comparison functions.

Pipeline
--------
1. common_average_reference() -- CAR each dataset (all simulated surface
   nodes / all recording electrodes) independently, since phi_e is only
   defined up to an additive constant per time instant and the two
   datasets use different, arbitrary reference conventions.
2. detect_fiducials_multichannel() -- one timestamp per beat, found from
   the RMS, ACROSS ALL CHANNELS, of the CAR'd signal's sample-to-sample
   derivative -- not the raw amplitude, and not just one picked channel.
   Multi-channel is more robust than picking on a single arbitrary
   electrode: ventricular depolarization produces a synchronized
   deflection across the whole torso, so its envelope has a clean peak
   at every beat regardless of what the picked channel's own morphology
   looks like. Using the derivative rather than raw amplitude matters
   separately: the QRS complex is a fast, narrow deflection while the
   T-wave is a slow, broad one whose absolute amplitude can exceed the
   QRS's at some electrodes even though it's the far slower feature.
   RMS-of-derivative suppresses the slow T-wave and highlights the fast
   QRS -- the same steep-slope logic Pan-Tompkins-style detectors rely
   on, but computed across the whole torso instead of one channel.
3. beat_window_from_rr() -- window size (pre, post), in SAMPLES of that
   dataset's own native units (frames for the simulation, samples for
   the recording), sized as a fraction of that dataset's own median
   beat-to-beat interval. No shared/known sample rate is required
   between the simulation (ms) and the recording (samples) -- each is
   windowed in its own units and only converted to a display axis
   afterward.
4. extract_average_beat() -- cut that window around every fiducial out of
   ONE channel's trace, NaN-pad whatever part of a beat's window runs
   off either end of the recording, and average with np.nanmean. A
   sample near the edge of the window that only one beat reaches is
   simply that beat's value there; no beat is discarded outright just
   because it's incomplete at one edge -- it contributes everywhere it
   has data and is silently absent (via NaN) elsewhere.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def common_average_reference(matrix: np.ndarray):
    """
    matrix: (n_channels, n_samples). Returns (car_matrix, removed_mean_t)
    where car_matrix = matrix - mean-across-channels-at-each-sample.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    mean_t = matrix.mean(axis=0)
    return matrix - mean_t[None, :], mean_t


def common_average_reference_using_channels(signal: np.ndarray, channel_idx: np.ndarray):
    """Common-average-reference `signal` (n_channels, n_frames), but compute
    the per-frame reference mean from only the rows in `channel_idx` instead
    of all rows. The reference is still subtracted from every channel, so
    the output shape/semantics match `common_average_reference()` -- this
    just changes which channels the mean is taken over.

    Used so the simulated BSPM's CAR is computed over the same electrode
    set as the recording's CAR (which, having only 128 electrode channels
    to begin with, is implicitly electrode-only already). See
    pick_and_compare_unipolar.py for why this matters for a single-point
    (unipolar) trace, and why pick_and_compare_bipolar.py doesn't need it
    (CAR cancels in an A-B difference regardless of which channels the
    reference is computed over).
    """
    channel_idx = np.asarray(channel_idx)
    reference = np.nanmean(signal[channel_idx, :], axis=0)
    return signal - reference[None, :], reference


def detect_fiducials_multichannel(
    car_matrix: np.ndarray,
    min_distance: int,
    prominence_frac: float = 0.35,
    verbose: bool = True,
    label: str = "",
):
    """
    car_matrix: (n_channels, n_samples), already common-average-referenced.
    min_distance: minimum allowed spacing between fiducials, in SAMPLES of
        this matrix's own time axis.
    prominence_frac: required peak prominence as a fraction of the RMS
        envelope's own dynamic range. Loosen (lower) if beats are missed,
        tighten (raise) if extra spurious peaks are detected.

    Returns (fiducials, rms_deriv) -- fiducials are sample indices INTO
    car_matrix's time axis (n_samples), one per detected beat.
    """
    rms_deriv = np.sqrt(np.mean(np.diff(car_matrix, axis=1) ** 2, axis=0))  # (n_samples-1,)
    dynamic_range = np.nanmax(rms_deriv) - np.nanmin(rms_deriv)
    prominence = prominence_frac * dynamic_range
    peaks, _ = find_peaks(rms_deriv, distance=min_distance, prominence=prominence)
    # np.diff shortens the array by 1 sample; shift +1 so a fiducial lands
    # ON the steepest post-derivative sample, right at the QRS upstroke.
    fiducials = peaks + 1
    if verbose:
        tag = f" ({label})" if label else ""
        print(f"    [detect_fiducials_multichannel{tag}] {len(fiducials)} beat(s) found")
    return fiducials, rms_deriv


def beat_window_from_rr(
    fiducials: np.ndarray,
    pre_frac: float,
    post_frac: float,
    fallback_length: int,
    verbose: bool = True,
    label: str = "",
):
    """
    Window size (pre, post), in samples, as a fraction of the median
    beat-to-beat (RR) interval -- unit-agnostic, so it works whether
    `fiducials` are frame indices (simulation) or sample indices
    (recording). With fewer than 2 beats there's no RR interval to
    measure, so this falls back to half of `fallback_length` (typically
    the channel's own length) as a sensible default window.
    """
    if len(fiducials) >= 2:
        rr = np.median(np.diff(fiducials))
    else:
        rr = fallback_length / 2.0
    pre = int(round(pre_frac * rr))
    post = int(round(post_frac * rr))
    if verbose:
        tag = f" ({label})" if label else ""
        print(f"    [beat_window_from_rr{tag}] pre={pre} post={post} samples, "
              f"from median RR={rr:.1f} samples")
    return pre, post, rr


def extract_average_beat(signal: np.ndarray, fiducials: np.ndarray, pre: int, post: int):
    """
    Cuts a window of `pre` samples before and `post` samples after each
    fiducial out of `signal`, NaN-pads whatever part of the window falls
    outside the recording, and averages with np.nanmean so partial/edge
    beats still contribute wherever they actually have data instead of
    being dropped outright.

    Returns
    -------
    avg    : (pre+post,) nan-averaged beat waveform
    count  : (pre+post,) number of beats contributing at each sample
    beats  : (n_beats, pre+post) individual beat windows, NaN where missing
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = pre + post
    beats = np.full((len(fiducials), n), np.nan)
    for i, f in enumerate(fiducials):
        start, end = f - pre, f + post
        seg_start, seg_end = max(start, 0), min(end, len(signal))
        if seg_start >= seg_end:
            continue  # this fiducial's window is entirely outside the recording
        out_start = seg_start - start
        out_end = out_start + (seg_end - seg_start)
        beats[i, out_start:out_end] = signal[seg_start:seg_end]
    count = np.sum(~np.isnan(beats), axis=0)
    avg = np.nanmean(beats, axis=0)
    return avg, count, beats


def plot_average_beat_panel(
    ax_top, ax_bottom, t_rel, avg, count, beats,
    color="black", label="average beat", title="",
):
    """Draws one dataset's average-beat panel (top: beats + average, bottom:
    per-sample beat-count coverage) onto a pair of existing matplotlib Axes."""
    for b in beats:
        ax_top.plot(t_rel, b, color=color, alpha=0.25, linewidth=0.9)
    ax_top.plot(t_rel, avg, color=color, linewidth=2.0, label=label)
    ax_top.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax_top.axhline(0, color="gray", linewidth=0.6, linestyle=":")
    if title:
        ax_top.set_title(title)
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc="upper right", fontsize=8)

    ax_bottom.fill_between(t_rel, count, step="mid", color=color, alpha=0.5)
    ax_bottom.set_ylabel("# beats\ncontributing")
    ax_bottom.set_ylim(0, max(int(count.max()), 1) + 0.5)
    ax_bottom.grid(True, alpha=0.3)
