"""
Synthetic OTDR Fault Dataset Generator
=======================================

Generates labeled synthetic OTDR traces for training/evaluating fault
detection & localization models (CNN/LSTM/Transformer/etc).

Physics model
-------------
- Rayleigh backscatter baseline decays linearly in dB with distance.
- Reflective events (fiber_cut, pc_connector, reflector) add a
  Gaussian-shaped reflectance spike on top of the baseline.
- Non-reflective events (bad_splice, bending, dirty_connector, tapping) add
  a downward step loss (dB) applied to everything beyond the event distance.
- Compound events: two or more events placed close enough that their
  signatures overlap (deliberately included -- this is under-represented in
  the public literature/datasets).
- Additive noise (white or mildly colored/pink) is added in the dB domain
  and clamped to a detector noise floor, scaled to hit a target SNR.

Reflectance convention (IMPORTANT -- corrected)
------------------------------------------------
OTDR "reflectance" is a return-loss ratio (reflected power / incident
power), so by definition it is always <= 0 dB -- there is no such thing as
positive reflectance. Realistic discrete-event reflectance values (per
standard OTDR engineering references):

    UPC connector (properly mated)         : -50 to -55 dB
    APC connector                          : -60 to -65 dB (best ~ -70 dB)
    Fusion splice                          : <= -60 dB (effectively non-reflective)
    Mechanical splice (index-matching gel) : -40 to -55 dB
    Unmated / open PC connector,
      perpendicular cleave (glass-air,
      ~3.5-4% Fresnel)                     : ~ -14 dB (poorly mated as bad as -11 dB)
    Angled cleave / shattered break        : much weaker, often <= -60 dB

The most reflective realistic discrete event is therefore about -11 to
-14 dB (an unmated/poorly-mated PC connector or a clean perpendicular
break); nothing in a real fiber plant produces positive-dB reflectance.

Because the *labeled* reflectance (return loss relative to the incident
pulse) and the *visible spike height above the backscatter trace* are not
the same quantity in real OTDR physics (a weak reflectance can still be
visually prominent because it is compared against the much weaker
distributed backscatter level, not the incident pulse), this generator
keeps them explicitly separate:

  - `reflectance_db` : the physically correct, negative-dB ground-truth
    label stored in the dataset (what a characterization model should
    learn to regress).
  - internal `spike_db` : a derived, always-non-negative quantity used only
    to render a visible spike in the synthetic trace, computed from
    `reflectance_db` via a fixed reference floor and gain (see
    `REFERENCE_FLOOR_DB` / `SPIKE_GAIN` below). This mirrors the real
    instrument fact that reflectance is reported relative to the incident
    pulse, while the spike you actually see on the trace is relative to
    the (much lower) backscatter level.

Output
------
1. `otdr_raw/<sample_id>.npy`      -- full-resolution raw trace (float32)
2. `OTDR_dataset.csv`              -- one row per sample:
      sample_id, snr_db, fiber_length_km, n_events, class,
      location_km, location_norm, reflectance_db, loss_db,
      P1..P30   (30-point normalized/downsampled trace, matches the
                 widely-used public "OTDR_Data" format for drop-in
                 compatibility with existing baselines)
3. `otdr_compound/<sample_id>.npy` + rows in `OTDR_compound_dataset.csv`
      -- multi-event / overlapping-event traces (own schema, since these
         are inherently multi-label)

Usage
-----
    python otdr_dataset_generator.py --n_samples 5000 --out_dir ./output

Everything is seeded for reproducibility; re-run with --seed to change draw.
"""

import argparse
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Fault taxonomy (matches the 8-class scheme used by the widely-reused public
# "OTDR_Data" benchmark, for compatibility with existing baselines)
# ---------------------------------------------------------------------------
CLASSES = {
    0: "normal",
    1: "fiber_tapping",
    2: "bad_splice",
    3: "bending",
    4: "dirty_connector",
    5: "fiber_cut",
    6: "pc_connector",
    7: "reflector",
}
REFLECTIVE_CLASSES = {5, 6, 7}      # produce a Fresnel spike
NONREFLECTIVE_CLASSES = {1, 2, 3, 4}  # produce only a step loss

# Typical physical ranges (edit to match your target fiber plant)
ALPHA_DB_PER_KM = (0.19, 0.25)      # attenuation coefficient range
FIBER_LENGTH_KM = (5.0, 40.0)

# --- Reflectance (dB), CORRECTED: always negative (return-loss convention) ---
# Class 5 (fiber_cut) is handled specially in draw_reflectance() below,
# since a real break is bimodal: a clean perpendicular cleave reflects
# strongly, while an angled/shattered break reflects weakly or not at all.
REFLECTANCE_DB = {
    6: (-14.0, -11.0),   # unmated/open PC connector, or poorly mated PC
    7: (-55.0, -40.0),   # generic reflective fault (e.g. mechanical splice w/ gel)
}
FIBER_CUT_CLEAN_DB = (-20.0, -14.0)   # clean perpendicular break / open end
FIBER_CUT_ANGLED_DB = (-70.0, -52.0)  # angled / shattered break -- weak reflection
FIBER_CUT_CLEAN_PROB = 0.5            # fraction of cuts that are "clean" perpendicular

# Reference floor and gain used ONLY to render a visible spike from a
# physically-correct (negative) reflectance label -- see module docstring.
REFERENCE_FLOOR_DB = -70.0   # reflectance at/below this is treated as invisible
SPIKE_GAIN = 0.542           # tuned so -11 dB (strongest realistic case) -> ~32 dB spike

LOSS_DB = {                         # (min, max) induced loss by class, dB
    1: (0.05, 0.5),                 # tapping -> very small, easy to miss
    2: (0.3, 3.0),                  # bad splice
    3: (0.5, 6.0),                  # bending
    4: (0.5, 4.0),                  # dirty connector
    # 5 (fiber_cut) is NOT a partial loss -- see apply_fiber_end() below.
    # There is no fiber beyond a complete break, so it cannot be modeled as
    # "baseline minus N dB"; it is a hard termination to the noise floor.
    6: (0.2, 1.5),                  # PC connector residual insertion loss
    7: (0.1, 1.0),                  # reflector residual loss
}
PULSE_WIDTH_KM = 0.02               # ~ spatial extent of the launch pulse response
NOISE_FLOOR_DB = -45.0              # detector noise floor relative to launch power


def db_to_lin(x_db):
    return 10.0 ** (x_db / 10.0)


def lin_to_db(x_lin, eps=1e-12):
    return 10.0 * np.log10(np.maximum(x_lin, eps))


def make_baseline(distance_km, alpha_db_per_km, p0_db=0.0):
    """Rayleigh backscatter baseline: linear decay in dB (round-trip)."""
    return p0_db - 2.0 * alpha_db_per_km * distance_km


def draw_reflectance(event_class, rng):
    """Draw a physically-correct (negative dB) reflectance value for a
    reflective event class. fiber_cut is bimodal: clean perpendicular
    breaks reflect strongly; angled/shattered breaks reflect weakly."""
    if event_class == 5:
        if rng.random() < FIBER_CUT_CLEAN_PROB:
            return rng.uniform(*FIBER_CUT_CLEAN_DB)
        else:
            return rng.uniform(*FIBER_CUT_ANGLED_DB)
    lo, hi = REFLECTANCE_DB[event_class]
    return rng.uniform(lo, hi)


def reflectance_to_spike_db(reflectance_db):
    """Map a physical (negative-dB) reflectance label to a non-negative
    'spike height above backscatter' used only for rendering the trace.
    See module docstring for why these are different quantities."""
    return max(0.0, SPIKE_GAIN * (reflectance_db - REFERENCE_FLOOR_DB))


def add_reflective_spike(trace_db, distance_km, d0_km, reflectance_db, pulse_width_km):
    """Superimpose a Gaussian-shaped Fresnel reflection spike at d0.
    `reflectance_db` is the physical (negative) label; internally converted
    to a rendering-only spike height via reflectance_to_spike_db().

    FIX (absolute vs. relative spike height, see Section VI-A validation):
    the previous version computed `spike_lin = db_to_lin(spike_db)`, treating
    spike_db as an absolute linear power referenced to 0 dB launch power --
    the same reference the backscatter baseline itself uses. This is only a
    good approximation of "height above local baseline" when spike_db is
    large enough to dominate the local baseline's linear power outright
    (true for most pc_connector/reflector draws). For small spike_db values
    (the low end of fiber_cut's angled/shattered-break range, near the
    reference floor), db_to_lin(spike_db) is close to 1.0 -- an absolute
    power near the *launch* level -- which can be orders of magnitude larger
    than a heavily attenuated local baseline far down the fiber, producing a
    rendered bump many dB taller than the intended label.

    Corrected model: build the spike's target power level as the local
    baseline (interpolated from the trace as it stands before this event)
    plus the intended height, exactly as Eqs. (2)-(4) specify, so the spike
    is always referenced to where it actually sits on the trace rather than
    to the fixed 0 dB launch reference.
    """
    spike_db = reflectance_to_spike_db(reflectance_db)
    local_baseline_db = np.interp(d0_km, distance_km, trace_db)
    spike_lin = db_to_lin(local_baseline_db + spike_db) * np.exp(
        -0.5 * ((distance_km - d0_km) / (pulse_width_km / 2.355)) ** 2
    )
    trace_lin = db_to_lin(trace_db) + spike_lin
    return lin_to_db(trace_lin)


def add_step_loss(trace_db, distance_km, d0_km, loss_db):
    """Apply a downward step loss to everything beyond d0 (non-reflective event)."""
    out = trace_db.copy()
    out[distance_km >= d0_km] -= loss_db
    return out


def apply_fiber_end(trace_db, distance_km, d0_km, pulse_width_km=PULSE_WIDTH_KM, margin_sigmas=4.0):
    """Complete fiber termination at d0 (fiber_cut / fiber end).

    Physically wrong model (previous version): subtract a large-but-finite
    step loss (20-60 dB) and let the baseline continue decaying beneath it.
    That implies fiber -- and therefore backscatter -- still exists beyond
    the break, just heavily attenuated. It does not.

    Correct model: beyond a complete break there is no fiber left to
    backscatter from at all, so the trace drops directly to the detector's
    noise floor with no further structure. This function overwrites
    everything beyond the break with the flat noise floor (noise is added on
    top of this later by add_noise, giving the familiar flat, noisy tail
    seen on real traces after a break/fiber end).

    FIX (grid-resolution edge case, see Section VI-A validation): the
    Fresnel reflection at the break is not an infinitely narrow spike -- it
    is smeared over a few standard deviations of the pulse width by the
    instrument response, exactly like every other reflective event in this
    generator (see add_reflective_spike()). The previous version hard-cut
    at `distance_km >= d0_km`, which is the exact center of that reflection
    -- on coarse distance grids (long fibers over a fixed sample count) this
    could clip the still-rising near side of the pulse before it reached
    its intended peak, occasionally producing a realized spike height below
    the local baseline. The cutoff is now placed `margin_sigmas` standard
    deviations beyond d0, letting the full reflection pulse render (as a
    real instrument would show it) before the trace drops to the noise
    floor.
    """
    out = trace_db.copy()
    sigma_km = pulse_width_km / 2.355
    cutoff_km = d0_km + margin_sigmas * sigma_km
    out[distance_km >= cutoff_km] = NOISE_FLOOR_DB
    return out


def add_noise(trace_db, target_snr_db, rng, colored=False):
    """Add Gaussian noise directly in the dB domain (avoids log-domain blow-up
    near zero linear power) and clamp to a detector noise floor, matching the
    flat noisy tail seen on real OTDR traces once backscatter drops below the
    instrument's sensitivity."""
    sigma_db = np.clip(3.0 * 10.0 ** (-(target_snr_db) / 20.0), 0.02, 6.0)
    noise = rng.normal(0, sigma_db, size=trace_db.shape)
    if colored:
        pink = np.cumsum(noise)
        pink -= np.mean(pink)
        pink = pink / (np.std(pink) + 1e-9) * np.std(noise)
        noise = 0.5 * noise + 0.5 * pink
    out = trace_db + noise
    floor_noise = rng.normal(0, sigma_db, size=trace_db.shape)
    out = np.where(trace_db <= NOISE_FLOOR_DB + 3 * sigma_db,
                    NOISE_FLOOR_DB + floor_noise, out)
    return out


def downsample_normalize(trace_db, n_points=30):
    """Match the public 'OTDR_Data' format: fixed-length normalized sequence."""
    idx = np.linspace(0, len(trace_db) - 1, n_points).astype(int)
    seq = trace_db[idx]
    seq = (seq - seq.min()) / (seq.max() - seq.min() + 1e-9)
    return seq


def generate_single_sample(rng, n_points_raw=2000, snr_range=(-5, 20), event_class=None):
    fiber_len = rng.uniform(*FIBER_LENGTH_KM)
    alpha = rng.uniform(*ALPHA_DB_PER_KM)
    distance = np.linspace(0, fiber_len, n_points_raw)
    trace = make_baseline(distance, alpha)

    if event_class is None:
        event_class = rng.integers(0, 8)

    location_km, reflectance_db, loss_db = np.nan, np.nan, np.nan

    if event_class != 0:
        location_km = rng.uniform(0.05 * fiber_len, 0.95 * fiber_len)

        if event_class == 5:
            # fiber_cut / fiber end: reflective spike (strong if clean
            # perpendicular break, weak/absent if angled or shattered --
            # handled by draw_reflectance's bimodal draw), then a hard
            # termination -- no fiber, no backscatter, beyond this point.
            reflectance_db = draw_reflectance(event_class, rng)
            trace = add_reflective_spike(trace, distance, location_km, reflectance_db, PULSE_WIDTH_KM)
            trace = apply_fiber_end(trace, distance, location_km, PULSE_WIDTH_KM)
            # "loss_db" here is a derived, deterministic quantity: how far
            # the pre-break baseline had to fall to reach the noise floor --
            # i.e. the effective total loss an instrument would report for a
            # complete break -- rather than a randomly drawn partial loss.
            loss_db = make_baseline(np.array([location_km]), alpha)[0] - NOISE_FLOOR_DB

        elif event_class in REFLECTIVE_CLASSES:
            reflectance_db = draw_reflectance(event_class, rng)
            trace = add_reflective_spike(trace, distance, location_km, reflectance_db, PULSE_WIDTH_KM)
            if event_class in LOSS_DB:
                loss_db = rng.uniform(*LOSS_DB[event_class])
                trace = add_step_loss(trace, distance, location_km, loss_db)

        elif event_class in NONREFLECTIVE_CLASSES:
            loss_db = rng.uniform(*LOSS_DB[event_class])
            trace = add_step_loss(trace, distance, location_km, loss_db)

    snr_db = rng.uniform(*snr_range)
    trace_noisy = add_noise(trace, snr_db, rng)

    return {
        "trace_raw": trace_noisy.astype(np.float32),
        "distance_km": distance.astype(np.float32),
        "fiber_length_km": fiber_len,
        "snr_db": snr_db,
        "class": event_class,
        "class_name": CLASSES[event_class],
        "location_km": location_km,
        "location_norm": (location_km / fiber_len) if not np.isnan(location_km) else np.nan,
        "reflectance_db": reflectance_db,
        "loss_db": loss_db,
    }


def generate_compound_sample(rng, n_points_raw=2000, snr_range=(-5, 20), n_events=2):
    """Multiple (possibly overlapping) events on one trace -- the compound/
    multi-fault case the public literature under-represents."""
    fiber_len = rng.uniform(*FIBER_LENGTH_KM)
    alpha = rng.uniform(*ALPHA_DB_PER_KM)
    distance = np.linspace(0, fiber_len, n_points_raw)
    trace = make_baseline(distance, alpha)

    events = []
    anchor = rng.uniform(0.1 * fiber_len, 0.6 * fiber_len)
    for i in range(n_events):
        if i == 0:
            loc = anchor
        else:
            overlap_draw = rng.random()
            if overlap_draw < 0.4:
                loc = anchor + rng.uniform(-3, 3) * PULSE_WIDTH_KM  # overlapping
            else:
                loc = rng.uniform(0.1 * fiber_len, 0.9 * fiber_len)  # separated
        loc = float(np.clip(loc, 0.02 * fiber_len, 0.98 * fiber_len))

        # fiber_cut (class 5) is deliberately excluded from the compound
        # pool: a complete break means no fiber -- and therefore no further
        # events -- can exist beyond it, which contradicts the premise of a
        # "compound" trace (multiple faults coexisting on a continuous
        # fiber). See apply_fiber_end() / generate_single_sample() for how
        # fiber_cut is modeled on its own.
        cls = int(rng.choice([1, 2, 3, 4, 6, 7]))
        refl, loss = np.nan, np.nan
        if cls in REFLECTIVE_CLASSES:
            refl = draw_reflectance(cls, rng)
            trace = add_reflective_spike(trace, distance, loc, refl, PULSE_WIDTH_KM)
        if cls in LOSS_DB:
            loss = rng.uniform(*LOSS_DB[cls])
            trace = add_step_loss(trace, distance, loc, loss)

        events.append({"class": cls, "class_name": CLASSES[cls],
                        "location_km": loc, "reflectance_db": refl, "loss_db": loss})

    snr_db = rng.uniform(*snr_range)
    trace_noisy = add_noise(trace, snr_db, rng)

    return {
        "trace_raw": trace_noisy.astype(np.float32),
        "distance_km": distance.astype(np.float32),
        "fiber_length_km": fiber_len,
        "snr_db": snr_db,
        "events": events,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_samples", type=int, default=5000)
    ap.add_argument("--n_compound", type=int, default=500)
    ap.add_argument("--snr_min", type=float, default=-5.0)
    ap.add_argument("--snr_max", type=float, default=20.0)
    ap.add_argument("--n_points_raw", type=int, default=2000)
    ap.add_argument("--n_points_norm", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="./otdr_synth_output")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    raw_dir = os.path.join(args.out_dir, "otdr_raw")
    compound_dir = os.path.join(args.out_dir, "otdr_compound")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(compound_dir, exist_ok=True)

    # ---- single-event dataset (drop-in compatible with public OTDR_Data format) ----
    rows = []
    for i in range(args.n_samples):
        s = generate_single_sample(rng, args.n_points_raw, (args.snr_min, args.snr_max))
        sample_id = f"sample_{i:06d}"
        np.save(os.path.join(raw_dir, f"{sample_id}.npy"), s["trace_raw"])
        seq30 = downsample_normalize(s["trace_raw"], args.n_points_norm)
        row = {
            "sample_id": sample_id,
            "snr_db": s["snr_db"],
            "fiber_length_km": s["fiber_length_km"],
            "class": s["class"],
            "class_name": s["class_name"],
            "location_km": s["location_km"],
            "location_norm": s["location_norm"],
            "reflectance_db": s["reflectance_db"],
            "loss_db": s["loss_db"],
        }
        row.update({f"P{j+1}": seq30[j] for j in range(args.n_points_norm)})
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "OTDR_dataset.csv"), index=False)

    # ---- compound / overlapping-event dataset ----
    crows = []
    for i in range(args.n_compound):
        n_ev = rng.integers(2, 4)
        s = generate_compound_sample(rng, args.n_points_raw, (args.snr_min, args.snr_max), n_events=n_ev)
        sample_id = f"compound_{i:05d}"
        np.save(os.path.join(compound_dir, f"{sample_id}.npy"), s["trace_raw"])
        for ev in s["events"]:
            crows.append({
                "sample_id": sample_id,
                "snr_db": s["snr_db"],
                "fiber_length_km": s["fiber_length_km"],
                **ev,
            })
    cdf = pd.DataFrame(crows)
    cdf.to_csv(os.path.join(args.out_dir, "OTDR_compound_dataset.csv"), index=False)

    print(f"Wrote {len(df)} single-event samples -> {args.out_dir}/OTDR_dataset.csv")
    print(f"Wrote {cdf['sample_id'].nunique()} compound samples "
          f"({len(cdf)} event rows) -> {args.out_dir}/OTDR_compound_dataset.csv")
    print(f"Raw traces saved under {raw_dir}/ and {compound_dir}/")


if __name__ == "__main__":
    main()
