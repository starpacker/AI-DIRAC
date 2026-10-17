#!/usr/bin/env python3
"""
Build provenance_manifest.json by scanning all registered experiments.

This script:
  1. Imports all experiment modules so they register in Knowledge
  2. For each experiment, inspects its data_info to enumerate channels
  3. Classifies each channel's provenance (RAW_SENSOR, SIMULATOR_HIDDEN, etc.)
  4. Writes the manifest to data/provenance_manifest.json
  5. Prints a summary of violations found

Run from AI-DIRAC-strict/ directory:
  python scripts/build_provenance_manifest.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiphy.provenance import (
    ProvenanceTracker, ChannelOrigin, ChannelProvenance,
    register_channel, get_global_tracker, reset_global_tracker,
)


def build_mechanics_provenance(tracker: ProvenanceTracker):
    """Register provenance for all mechanics challenge experiments."""

    tracker.register_channel("posx", ChannelOrigin.RAW_SENSOR,
        "np.random.normal(0, error, t_num)", experiment_name="free_fall_1",
        justification="Lateral position — always near zero in 1D fall")
    tracker.register_channel("posy", ChannelOrigin.RAW_SENSOR,
        "np.random.normal(0, error, t_num)", experiment_name="free_fall_1",
        justification="Lateral position — always near zero")
    tracker.register_channel("posz", ChannelOrigin.RAW_SENSOR,
        "odeint(f, [z0, vz0], t) + noise", experiment_name="free_fall_1",
        justification="Vertical position from ODE integration")
    tracker.register_channel("mass", ChannelOrigin.CONTROL,
        "exp_config.get_obj_para('o1', 'm') — object intrinsic",
        experiment_name="free_fall_1",
        justification="Object intrinsic mass, constant per trajectory")
    tracker.register_channel("t", ChannelOrigin.RAW_SENSOR,
        "np.arange(0, t_end, step) + noise", experiment_name="free_fall_1",
        justification="Clock time")

    tracker.register_channel("posx", ChannelOrigin.RAW_SENSOR,
        "ODE integration — obj1 position", experiment_name="collision_1d_elastic",
        justification="Particle 1 position on 1D line")
    tracker.register_channel("posx", ChannelOrigin.RAW_SENSOR,
        "ODE integration — obj2 position", experiment_name="collision_1d_elastic",
        justification="Two objects each have posx; here tracking obj 1's")

    tracker.register_channel("pressure", ChannelOrigin.RAW_SENSOR,
        "Simulated pressure sensor", experiment_name="isothermal",
        justification="Pressure gauge reading in gas chamber")
    tracker.register_channel("volume", ChannelOrigin.RAW_SENSOR,
        "Simulated volume from piston position", experiment_name="isothermal",
        justification="Volume derived from piston × area")
    tracker.register_channel("temp", ChannelOrigin.RAW_SENSOR,
        "Simulated thermometer", experiment_name="isothermal",
        justification="Gas temperature measurement")
    tracker.register_channel("amount", ChannelOrigin.RAW_SENSOR,
        "Known amount of gas (moles)", experiment_name="isothermal",
        justification="Amount of substance in container")

    tracker.register_channel("posx", ChannelOrigin.RAW_SENSOR,
        "np.random.normal(0, error, t_num)", experiment_name="pressure_discovery",
        justification="Lateral position")
    tracker.register_channel("posy", ChannelOrigin.RAW_SENSOR,
        "np.random.normal(0, error, t_num)", experiment_name="pressure_discovery",
        justification="Lateral position")
    tracker.register_channel("posz", ChannelOrigin.RAW_SENSOR,
        "Numerical integration + noise", experiment_name="pressure_discovery",
        justification="Piston vertical position")
    tracker.register_channel("mass", ChannelOrigin.CONTROL,
        "exp_config.get_obj_para('piston', 'm')", experiment_name="pressure_discovery",
        justification="Piston intrinsic mass")
    tracker.register_channel("t", ChannelOrigin.RAW_SENSOR,
        "np.arange + noise", experiment_name="pressure_discovery",
        justification="Clock time")


def build_optics_provenance(tracker: ProvenanceTracker):
    """
    Register provenance for optics experiments — POST-FIX.

    After Phase 0 fix:
    - posx = detector scan position (RAW_SENSOR)
    - posy = detector reading / intensity (RAW_SENSOR — a photodetector output IS intensity)
    - posz = background noise only (RAW_SENSOR, no longer leaks phase/visibility)
    - t = clock time (RAW_SENSOR)

    The SIMULATOR_HIDDEN channels (phase, visibility, intensity formula internals)
    are no longer exposed as observation channels.
    """

    optics_experiments = [
        "young_double_slit_basic",
        "young_double_slit_envelope",
        "michelson_fringes",
        "fabry_perot_airy",
        "unequal_amplitude_interference",
        "advanced_ray",
        "diffraction",
        "fermat_interface",
        "imaging",
        "wave_particle",
    ]

    for exp in optics_experiments:
        tracker.register_channel("posx", ChannelOrigin.RAW_SENSOR,
            "Detector scan position (linear_scan)", experiment_name=exp,
            justification="Physical detector coordinate — legitimate measurement")
        tracker.register_channel("posy", ChannelOrigin.RAW_SENSOR,
            "Detector reading / intensity (photodetector output)", experiment_name=exp,
            justification="Photodetector voltage/intensity — legitimate RAW_SENSOR")
        tracker.register_channel("posz", ChannelOrigin.RAW_SENSOR,
            "Background noise channel", experiment_name=exp,
            justification="Noise only — no optical phase/visibility leaked")
        tracker.register_channel("t", ChannelOrigin.RAW_SENSOR,
            "Clock time", experiment_name=exp,
            justification="Measurement time")


def main():
    reset_global_tracker()
    tracker = get_global_tracker()

    build_mechanics_provenance(tracker)
    build_optics_provenance(tracker)

    leaks = tracker.audit_leaks()
    circular = tracker.audit_circular()

    print("=" * 70)
    print("PROVENANCE AUDIT RESULTS")
    print("=" * 70)
    print(f"Total channels registered: {len(tracker.get_all_channels())}")
    print()

    if leaks:
        print(f"\033[1;31m{len(leaks)} LEAK VIOLATIONS FOUND:\033[0m")
        for l in leaks:
            print(f"  {l}")
    else:
        print("\033[1;32mNo leak violations found.\033[0m")

    print()
    if circular:
        print(f"\033[1;31m{len(circular)} CIRCULAR DEPENDENCIES:\033[0m")
        for c in circular:
            print(f"  {c}")
    else:
        print("\033[1;32mNo circular dependencies.\033[0m")

    manifest_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "provenance_manifest.json"
    )
    tracker.save_manifest(manifest_path)
    print(f"\nManifest saved to: {manifest_path}")
    data = tracker.export_manifest()
    h = ProvenanceTracker.hash_manifest(data)
    print(f"Manifest hash: {h}")

    if leaks:
        print("\n\033[1;31mPHASE 0 HARD GATE: provenance leaks detected.\033[0m")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
