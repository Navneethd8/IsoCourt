"""Court-map sampling must not retain full-resolution frames."""
import weakref

import cv2
import numpy as np

from api import state
from api.clip_analysis import (
    _center_line_x,
    _court_floor_box,
    _court_reference,
    _foot_on_court,
    _horizontal_court_lines,
    _hough_segments,
    _place_rally_feet,
    sample_rally_feet,
)


class _AnklePose:
    """Pose stub. Does not keep the frame."""

    def process_frame(self, frame):
        return None

    def get_landmarks_as_list(self, _detected):
        landmarks = []
        for index in range(33):
            landmarks.append({"id": index, "x": 0.5, "y": 0.82, "z": 0.0, "visibility": 0.95})
        landmarks[27]["x"] = 0.46
        landmarks[28]["x"] = 0.54
        return [landmarks]


def _write_court_clip(path, n_frames, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    assert writer.isOpened()
    for _ in range(n_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (40, 160, 40)
        for y_frac in (0.50, 0.62, 0.74, 0.88):
            y = int(height * y_frac)
            cv2.line(frame, (0, y), (width - 1, y), (240, 240, 240), 2)
        writer.write(frame)
    writer.release()


def _legacy_sample(path):
    """Previous two-pass sampler, used only to lock court-map output."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
    step = max(1, int(round(fps / 8)))
    frames = []
    for idx in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append((idx, frame))
    cap.release()
    boxes = [box for box in (_court_floor_box(frame) for _, frame in frames) if box]
    if not frames or not boxes:
        return []
    points = []
    references = []
    centers = []
    for idx, frame in frames:
        box = _court_floor_box(frame)
        if box is None:
            continue
        detected = state.pose_estimator.process_frame(frame)
        people = state.pose_estimator.get_landmarks_as_list(detected)
        if not people:
            continue
        foot = _foot_on_court(people[0], frame, box)
        if not foot:
            continue
        reference = _court_reference(frame, foot["py"])
        if reference:
            references.append(reference)
        center = _center_line_x(frame)
        if center is not None:
            centers.append((center, frame.shape[1]))
        foot["frame"] = int(idx)
        points.append(foot)
    return _place_rally_feet(points, references, centers)


class _TrackingCapture:
    def __init__(self, real_capture, path):
        self._cap = real_capture(path)
        self._refs = []
        self.peak_held = 0
        self.reads = 0

    def get(self, prop):
        return self._cap.get(prop)

    def set(self, prop, value):
        return self._cap.set(prop, value)

    def read(self):
        held = sum(1 for ref in self._refs if ref() is not None)
        self.peak_held = max(self.peak_held, held)
        ok, frame = self._cap.read()
        self.reads += 1
        if ok and frame is not None:
            self._refs.append(weakref.ref(frame))
        return ok, frame

    def release(self):
        self._cap.release()

    def isOpened(self):
        return self._cap.isOpened()


def test_sample_rally_feet_matches_frame_buffering_output(tmp_path, monkeypatch):
    path = str(tmp_path / "court.mp4")
    _write_court_clip(path, n_frames=48, fps=24, width=320, height=180)
    monkeypatch.setattr(state, "pose_estimator", _AnklePose())
    fresh = sample_rally_feet(path)
    legacy = _legacy_sample(path)
    assert fresh
    assert fresh == legacy
    assert all(0.0 <= point["x"] <= 1.0 and 0.0 <= point["y"] <= 1.0 for point in fresh)


def test_sample_rally_feet_does_not_accumulate_frames(tmp_path, monkeypatch):
    path = str(tmp_path / "long-court.mp4")
    n_frames = 600
    fps = 25
    _write_court_clip(path, n_frames=n_frames, fps=fps, width=320, height=180)
    monkeypatch.setattr(state, "pose_estimator", _AnklePose())
    tracker = {}
    real_capture = cv2.VideoCapture

    def factory(source):
        cap = _TrackingCapture(real_capture, source)
        tracker["cap"] = cap
        return cap

    monkeypatch.setattr("api.clip_analysis.cv2.VideoCapture", factory)
    points = sample_rally_feet(path)
    cap = tracker["cap"]
    step = max(1, int(round(fps / 8)))
    sampled = len(range(0, n_frames, step))
    assert cap.reads >= sampled
    assert points
    # A retained-frame list would still be holding the earlier samples here.
    assert cap.peak_held <= 2
    assert cap.peak_held < sampled / 4


def test_hough_segments_use_opencv5_rows_and_accept_opencv4():
    segment = np.array([[0, 50, 159, 50], [3, 12, 40, 12]], dtype=np.int32)
    native = _hough_segments(segment)  # OpenCV 5: (N, 4)
    legacy = _hough_segments(segment.reshape(2, 1, 4))  # OpenCV 4: (N, 1, 4)
    assert native.shape == (2, 4)
    assert np.array_equal(native, segment)
    assert np.array_equal(legacy, segment)
    assert _hough_segments(None).shape == (0, 4)


def test_installed_opencv_houghlinesp_is_five_layout():
    assert cv2.__version__.split(".", 1)[0] == "5"
    image = np.zeros((80, 160), dtype=np.uint8)
    cv2.line(image, (0, 40), (159, 40), 255, 2)
    raw = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=20, minLineLength=40, maxLineGap=4)
    assert raw is not None
    assert raw.ndim == 2 and raw.shape[1] == 4
    frame = np.zeros((80, 160, 3), dtype=np.uint8)
    frame[:, :] = (40, 160, 40)
    cv2.line(frame, (0, 40), (159, 40), (255, 255, 255), 2)
    lines = _horizontal_court_lines(frame)
    assert lines
    assert any(abs(y - 40) <= 2 for y, _length in lines)


def test_sample_rally_feet_empty_without_pose_or_court(tmp_path, monkeypatch):
    path = str(tmp_path / "blank.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 10, (64, 64))
    for _ in range(10):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()
    monkeypatch.setattr(state, "pose_estimator", None)
    assert sample_rally_feet(path) == []
    monkeypatch.setattr(state, "pose_estimator", _AnklePose())
    assert sample_rally_feet(path) == []
