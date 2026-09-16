import unittest
from datetime import date, datetime, time, timedelta
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from qwip_explorer.core import (
    assets_to_frame,
    commits_to_frame,
    datasets_to_frame,
    extract_dataframes,
    filter_datasets_by_protocol,
    flatten_frame,
    heatmap_figure,
    line_figure,
    normalize_dataset_id,
    plottable_columns,
    timestamp_bounds,
)


class FakeAsset:
    serializer = "dataframe"
    fmt = "parquet"
    address = "/abc/data.parquet"
    storage = object()
    params = {}


class FakeDataset:
    id = "abc"
    timestamp = "2026-08-18T12:00:00"
    protocol = "Rabi 2D"
    comments = "test"
    sample_id = "sample"
    cooldown_id = "cooldown"
    user = "user"
    host = "host"
    config_db = "localhost/xonium"
    commit = "1234"
    version = "1"

    def assets(self):
        return ("data",)

    def __getitem__(self, name):
        return FakeAsset()


class CoreTests(unittest.TestCase):
    def test_normalizes_dataset_uuid(self):
        self.assertEqual(
            normalize_dataset_id("01a01754-d305-77e7-9f5d-f5e9a0c52422"),
            "01a01754d30577e79f5df5e9a0c52422",
        )
        with self.assertRaisesRegex(ValueError, "missing a leading"):
            normalize_dataset_id("1a01754-d305-77e7-9f5d-f5e9a0c52422")

    def test_dataset_and_asset_summaries(self):
        self.assertEqual(datasets_to_frame([FakeDataset()]).iloc[0]["protocol"], "Rabi 2D")
        self.assertEqual(assets_to_frame(FakeDataset()).iloc[0]["format"], "parquet")

    def test_filters_and_sorts_commit_history(self):
        commits = [
            SimpleNamespace(
                hash="a" * 32,
                committer="Other User",
                email="other@example.com",
                date="2026-08-20T12:00:00-07:00",
                message="Older commit",
            ),
            SimpleNamespace(
                hash="b" * 32,
                committer="Chuan-Hong Liu",
                email="chuanhongliu@berkeley.edu",
                date="2026-08-22T13:00:00-07:00",
                message="Newest matching commit",
            ),
            SimpleNamespace(
                hash="c" * 32,
                committer="Chuan-Hong Liu",
                email="chuanhongliu@berkeley.edu",
                date="2026-08-21T13:00:00-07:00",
                message="Older matching commit",
            ),
        ]

        frame = commits_to_frame(commits, "chuanhong", limit=1)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["message"], "Newest matching commit")
        self.assertEqual(frame.iloc[0]["short_hash"], "bbbbbbbb")
        self.assertEqual(frame.iloc[0]["commit_hash"], "b" * 32)

    def test_filters_protocol_case_insensitively(self):
        datasets = [FakeDataset(), SimpleNamespace(protocol="T1")]
        self.assertEqual(filter_datasets_by_protocol(datasets, "rabi"), [datasets[0]])

    def test_builds_timestamp_presets(self):
        now = datetime(2026, 8, 24, 13, 30, tzinfo=ZoneInfo("America/Los_Angeles"))
        start, end = timestamp_bounds("This month", now=now)
        self.assertEqual(start, datetime(2026, 8, 1, tzinfo=now.tzinfo))
        self.assertEqual(end, now)

        start, end = timestamp_bounds("Last 24 hours", now=now)
        self.assertEqual(start, now - timedelta(hours=24))
        self.assertEqual(end, now)

    def test_builds_and_validates_custom_timestamp_range(self):
        start, end = timestamp_bounds(
            "Custom range",
            start_date=date(2026, 8, 20),
            start_time=time(9, 15),
            end_date=date(2026, 8, 21),
            end_time=time(17, 45),
        )
        self.assertEqual(start.isoformat(), "2026-08-20T09:15:00-07:00")
        self.assertEqual(end.isoformat(), "2026-08-21T17:45:00-07:00")

        with self.assertRaisesRegex(ValueError, "start timestamp"):
            timestamp_bounds(
                "Custom range",
                start_date=date(2026, 8, 22),
                start_time=time(9),
                end_date=date(2026, 8, 21),
                end_time=time(17),
            )

    def test_extracts_nested_measurement_frames(self):
        frame = pd.DataFrame({"value": [1, 2]})
        result = {"Q1": SimpleNamespace(name="Q1", data=frame)}
        extracted = extract_dataframes(result)
        self.assertIn("Q1", extracted)
        pd.testing.assert_frame_equal(extracted["Q1"], frame)

    def test_flattens_two_dimensional_sweep(self):
        index = pd.MultiIndex.from_product([[0, 1], [10, 20]], names=["amplitude", "time"])
        frame = pd.DataFrame({"population": [0.1, 0.2, 0.3, 0.4]}, index=index)
        flat, axes = flatten_frame(frame)
        self.assertEqual(axes, ["amplitude", "time"])
        self.assertEqual(plottable_columns(flat, axes), ["population"])
        self.assertEqual(len(flat), 4)

    def test_builds_line_and_heatmap_figures(self):
        frame = pd.DataFrame(
            {
                "amplitude": [0, 0, 1, 1],
                "time": [10, 20, 10, 20],
                "value": np.array([1, 2, 3, 4], dtype=float),
            }
        )
        line = line_figure(frame, x="time", value="value", color="amplitude")
        heatmap = heatmap_figure(frame, x="time", y="amplitude", value="value")
        self.assertEqual(len(line.data), 2)
        self.assertEqual(heatmap.data[0].z.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
