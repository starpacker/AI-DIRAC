"""
Test provenance audit — verifies that:
  1. Channel origin types are correctly classified
  2. Leak detection correctly flags SIMULATOR_HIDDEN channels accessible to discovery
  3. ANSWER_TEMPLATE channels are flagged
  4. Circular dependency detection works
  5. Manifest export/import is round-trip consistent
"""

import sys
import os
import json
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiphy.provenance import (
    ProvenanceTracker, ChannelOrigin, ChannelProvenance,
    register_channel, get_global_tracker, reset_global_tracker,
    audit_all,
)


class TestProvenanceAudit(unittest.TestCase):

    def setUp(self):
        reset_global_tracker()

    def test_register_and_retrieve(self):
        """Channels can be registered and retrieved."""
        tracker = get_global_tracker()
        tracker.register_channel(
            "posx", ChannelOrigin.RAW_SENSOR, "odeint + noise",
            experiment_name="free_fall_1",
            justification="Position measurement")
        prov = tracker.get_channel("posx", "free_fall_1")
        self.assertIsNotNone(prov)
        self.assertEqual(prov.channel_name, "posx")
        self.assertEqual(prov.origin, ChannelOrigin.RAW_SENSOR)
        self.assertEqual(prov.experiment_name, "free_fall_1")

    def test_leak_detection_simulator_hidden(self):
        """SIMULATOR_HIDDEN channels accessible to discovery are flagged as leaks."""
        tracker = get_global_tracker()
        tracker.register_channel(
            "phase", ChannelOrigin.SIMULATOR_HIDDEN,
            "phase = 2*pi*d*x/(lam*L)",
            experiment_name="young_double_slit_basic",
            accessible_to_discovery=True,
            justification="This is a leak")
        leaks = tracker.audit_leaks()
        self.assertGreater(len(leaks), 0)
        self.assertTrue(any("phase" in l for l in leaks))
        self.assertTrue(any("LEAK" in l for l in leaks))

    def test_no_leak_when_hidden_not_accessible(self):
        """SIMULATOR_HIDDEN channels NOT accessible to discovery are not flagged."""
        tracker = get_global_tracker()
        tracker.register_channel(
            "phase", ChannelOrigin.SIMULATOR_HIDDEN,
            "phase computed internally",
            experiment_name="young_double_slit_basic",
            accessible_to_discovery=False,
            justification="Properly hidden")
        leaks = tracker.audit_leaks()
        self.assertEqual(len(leaks), 0)

    def test_answer_template_leak(self):
        """ANSWER_TEMPLATE channels accessible to discovery are flagged as leaks."""
        tracker = get_global_tracker()
        tracker.register_channel(
            "energy", ChannelOrigin.ANSWER_TEMPLATE,
            "kinetic + potential",
            experiment_name="free_fall_1",
            accessible_to_discovery=True,
            justification="Answer template leak")
        leaks = tracker.audit_leaks()
        self.assertTrue(any("energy" in l for l in leaks))

    def test_derived_channel_with_hidden_upstream(self):
        """DERIVED channels depending on SIMULATOR_HIDDEN upstream are flagged."""
        tracker = get_global_tracker()
        tracker.register_channel(
            "phase_raw", ChannelOrigin.SIMULATOR_HIDDEN,
            "internal phase computation",
            experiment_name="test_exp",
            accessible_to_discovery=False)
        tracker.register_channel(
            "derived_metric", ChannelOrigin.DERIVED,
            "some function of phase_raw",
            dependencies=["phase_raw"],
            experiment_name="test_exp",
            accessible_to_discovery=True,
            justification="Derived from hidden upstream")
        leaks = tracker.audit_leaks()
        self.assertTrue(any("derived_metric" in l for l in leaks))

    def test_circular_dependency_detection(self):
        """Circular dependencies are detected."""
        tracker = get_global_tracker()
        tracker.register_channel(
            "A", ChannelOrigin.RAW_SENSOR, "direct",
            dependencies=["B"], experiment_name="circ_test")
        tracker.register_channel(
            "B", ChannelOrigin.DERIVED, "from A",
            dependencies=["A"], experiment_name="circ_test")
        circular = tracker.audit_circular()
        self.assertGreater(len(circular), 0)

    def test_no_circular_when_acyclic(self):
        """Acyclic dependency graphs have no circular report."""
        tracker = get_global_tracker()
        tracker.register_channel(
            "C", ChannelOrigin.RAW_SENSOR, "direct",
            experiment_name="acyc_test")
        tracker.register_channel(
            "D", ChannelOrigin.DERIVED, "from C",
            dependencies=["C"], experiment_name="acyc_test")
        circular = tracker.audit_circular()
        self.assertEqual(len(circular), 0)

    def test_manifest_roundtrip(self):
        """Manifest export → import preserves all provenance data."""
        tracker = get_global_tracker()
        tracker.register_channel(
            "posx", ChannelOrigin.RAW_SENSOR, "odeint",
            experiment_name="free_fall_1",
            justification="Position")

        data = tracker.export_manifest()
        data_json = json.dumps(data)
        data2 = json.loads(data_json)

        # Save to temp file, reload
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
            tracker.save_manifest(tf.name)
            tmp_path = tf.name

        try:
            tracker2 = ProvenanceTracker.load_manifest(tmp_path)
            prov1 = tracker.get_channel("posx", "free_fall_1")
            prov2 = tracker2.get_channel("posx", "free_fall_1")
            self.assertIsNotNone(prov1)
            self.assertIsNotNone(prov2)
            self.assertEqual(prov1.channel_name, prov2.channel_name)
            self.assertEqual(prov1.origin, prov2.origin)
            self.assertEqual(prov1.generation_method, prov2.generation_method)
        finally:
            os.unlink(tmp_path)

    def test_hash_deterministic(self):
        """Manifest hash is deterministic for same content."""
        data = {"channels": [], "format_version": "1.0"}
        h1 = ProvenanceTracker.hash_manifest(data)
        h2 = ProvenanceTracker.hash_manifest(data)
        self.assertEqual(h1, h2)

    def test_hash_differs_for_different_content(self):
        """Manifest hash differs for different content."""
        h1 = ProvenanceTracker.hash_manifest({"channels": [], "format_version": "1.0"})
        h2 = ProvenanceTracker.hash_manifest({"channels": [{"name": "x"}], "format_version": "1.0"})
        self.assertNotEqual(h1, h2)

    def test_raw_sensor_not_flagged(self):
        """RAW_SENSOR channels are never flagged as leaks."""
        tracker = get_global_tracker()
        tracker.register_channel(
            "posx", ChannelOrigin.RAW_SENSOR, "measurement",
            experiment_name="test", accessible_to_discovery=True)
        tracker.register_channel(
            "t", ChannelOrigin.RAW_SENSOR, "clock",
            experiment_name="test", accessible_to_discovery=True)
        leaks = tracker.audit_leaks()
        self.assertEqual(len(leaks), 0)

    def test_control_not_flagged(self):
        """CONTROL parameters (mass, etc.) are not flagged as leaks."""
        tracker = get_global_tracker()
        tracker.register_channel(
            "mass", ChannelOrigin.CONTROL, "object parameter",
            experiment_name="test", accessible_to_discovery=True)
        leaks = tracker.audit_leaks()
        self.assertEqual(len(leaks), 0)

    def test_optics_simulator_leak_scenario(self):
        """Full optics scenario: phase and intensity exposed as posz/posy → all leaks detected."""
        tracker = get_global_tracker()
        tracker.register_channel("posx", ChannelOrigin.RAW_SENSOR,
            "scan position", experiment_name="optics_test")
        tracker.register_channel("posy", ChannelOrigin.SIMULATOR_HIDDEN,
            "intensity = cos(phase)", experiment_name="optics_test",
            accessible_to_discovery=True)
        tracker.register_channel("posz", ChannelOrigin.SIMULATOR_HIDDEN,
            "phase = 2*pi*x/lambda", experiment_name="optics_test",
            accessible_to_discovery=True)
        tracker.register_channel("t", ChannelOrigin.RAW_SENSOR,
            "time", experiment_name="optics_test")
        leaks = tracker.audit_leaks()
        self.assertEqual(len(leaks), 2)
        self.assertTrue(any("posy" in l for l in leaks))
        self.assertTrue(any("posz" in l for l in leaks))


if __name__ == "__main__":
    unittest.main()
