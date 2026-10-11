"""Tests for NorfairTracker when autotracking settings change at runtime.

The camera process receives onvif and autotracking config updates, so the
values get_tracker uses to pick a "static" or "ptz" tracker can change after
the per object type trackers were built. Disabling autotracking from the UI
used to raise a KeyError in match_and_update, which killed the camera's
tracking process until Frigate was restarted.
"""

import unittest
from unittest.mock import MagicMock

import numpy as np

from frigate.camera import PTZMetrics
from frigate.config import FrigateConfig
from frigate.track.norfair_tracker import NorfairTracker

CAMERA = "ptz_cam"
WIDTH = 1280
HEIGHT = 720


def _config(
    autotracking_enabled: bool, track: list[str] | None = None
) -> FrigateConfig:
    autotracking: dict = {
        "enabled": autotracking_enabled,
        "required_zones": ["zone"],
    }

    if track is not None:
        autotracking["track"] = track

    return FrigateConfig(
        **{
            "mqtt": {"enabled": False},
            "cameras": {
                CAMERA: {
                    "ffmpeg": {
                        "inputs": [
                            {"path": "rtsp://10.0.0.1:554/video", "roles": ["detect"]}
                        ]
                    },
                    "detect": {"width": WIDTH, "height": HEIGHT},
                    "zones": {"zone": {"coordinates": "0,0,1,0,1,1,0,1"}},
                    "onvif": {
                        "host": "10.0.0.1",
                        "autotracking": autotracking,
                    },
                }
            },
        }
    )


def _make_tracker(autotracking_enabled: bool) -> NorfairTracker:
    camera_config = _config(autotracking_enabled).cameras[CAMERA]
    tracker = NorfairTracker(camera_config, PTZMetrics())
    yuv_frame = np.zeros(camera_config.frame_shape_yuv, dtype=np.uint8)
    tracker.frame_manager = MagicMock()
    tracker.frame_manager.get.return_value = yuv_frame
    # no camera movement, so no coordinate transformations
    tracker.ptz_motion_estimator = MagicMock()
    tracker.ptz_motion_estimator.motion_estimator.return_value = None
    return tracker


def _apply_onvif_update(tracker: NorfairTracker, config: FrigateConfig) -> None:
    """Mirror CameraConfigUpdateSubscriber applying an onvif update."""
    tracker.camera_config.onvif = config.cameras[CAMERA].onvif


def _run_frames(
    tracker: NorfairTracker, start: float, count: int, label: str = "person"
) -> float:
    """Feed a stationary detection for count frames."""
    frame_time = start

    for _ in range(count):
        frame_time += 0.2
        box = (400, 200, 500, 500)
        tracker.match_and_update(
            "frame",
            frame_time,
            [(label, 0.9, box, 100 * 300, 100 / 300, (0, 0, 640, 640))],
        )

    return frame_time


class TestTrackerRuntimeAutotrackingChanges(unittest.TestCase):
    def test_disabling_autotracking_keeps_tracking(self) -> None:
        tracker = _make_tracker(autotracking_enabled=True)
        frame_time = _run_frames(tracker, 1000.0, 10)
        self.assertEqual(len(tracker.tracked_objects), 1)

        # saving onvif settings in the UI publishes a re-parsed config, so
        # enabled_in_config flips to False in the camera process
        _apply_onvif_update(tracker, _config(autotracking_enabled=False))
        _run_frames(tracker, frame_time, 10)

        self.assertIs(tracker.get_tracker("person"), tracker.default_tracker["static"])
        self.assertEqual(len(tracker.tracked_objects), 1)
        self.assertEqual(
            next(iter(tracker.tracked_objects.values()))["label"], "person"
        )

    def test_removing_person_from_autotracking_keeps_tracking(self) -> None:
        tracker = _make_tracker(autotracking_enabled=True)
        frame_time = _run_frames(tracker, 1000.0, 10)

        _apply_onvif_update(tracker, _config(autotracking_enabled=True, track=["car"]))
        _run_frames(tracker, frame_time, 10)

        self.assertNotIn("person", tracker.trackers)
        self.assertIs(tracker.get_tracker("person"), tracker.default_tracker["ptz"])
        self.assertEqual(len(tracker.tracked_objects), 1)

    def test_enabling_autotracking_uses_ptz_person_tracker(self) -> None:
        tracker = _make_tracker(autotracking_enabled=False)
        frame_time = _run_frames(tracker, 1000.0, 10)
        self.assertEqual(len(tracker.tracked_objects), 1)

        _apply_onvif_update(tracker, _config(autotracking_enabled=True))
        _run_frames(tracker, frame_time, 10)

        self.assertIs(tracker.get_tracker("person"), tracker.trackers["person"]["ptz"])
        self.assertEqual(len(tracker.tracked_objects), 1)

    def test_unchanged_onvif_update_keeps_existing_objects(self) -> None:
        tracker = _make_tracker(autotracking_enabled=True)
        frame_time = _run_frames(tracker, 1000.0, 10)
        object_ids = set(tracker.tracked_objects)

        # an onvif save that doesn't touch autotracking must not reset tracking
        _apply_onvif_update(tracker, _config(autotracking_enabled=True))
        _run_frames(tracker, frame_time, 5)

        self.assertEqual(set(tracker.tracked_objects), object_ids)


class TestTrackerDefaultMode(unittest.TestCase):
    def test_default_objects_use_ptz_default_tracker(self) -> None:
        # objects without their own tracker are matched by the default ptz
        # tracker on an autotracking camera, so get_tracker must return it
        tracker = _make_tracker(autotracking_enabled=True)
        _run_frames(tracker, 1000.0, 10, label="dog")

        self.assertIs(tracker.get_tracker("dog"), tracker.default_tracker["ptz"])
        self.assertEqual(len(tracker.tracked_objects), 1)
        obj = next(iter(tracker.tracked_objects.values()))
        # register found the norfair object, so the pre-initialization
        # detections were carried into the score history
        self.assertGreater(len(obj["score_history"]), 1)


if __name__ == "__main__":
    unittest.main()
