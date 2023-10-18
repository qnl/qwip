import pytest
import sqlalchemy as sa
from uuid6 import uuid7

from qwip.data.models import Asset, Dataset
from qwip.data.storage import HTTPStorageBackend
from qwip.processing.processors import IQResult


class TestDataset:
    @pytest.fixture
    def dataset(self):
        dataset = Dataset(
            user="qnl",
            sample_id="K210210",
            cooldown_id="SNB230929",
            comments="Test dataset",
        )

        return dataset

    def test_insert_select(self, session, models, dataset):
        session.add(dataset)
        session.flush()

        results = session.scalars(sa.select(Dataset)).all()
        assert len(results) == 1
        assert results == [dataset]

    def test_dataset_add(self, dataset, storage):
        plot = Asset(
            name="plot",
            storage=storage,
        )

        data = Asset(name="data", storage=storage, serializer="result")

        metadata = Asset(
            name="metadata",
            storage=storage,
        )

        dataset.add(plot)

        dataset.add([data, metadata])

        assert "plot" in dataset._assets
        assert "data" in dataset._assets
        assert "metadata" in dataset._assets

        assert plot.dataset_id == data.dataset_id == metadata.dataset_id == dataset.id

    def test_dataset_getitem(self, dataset, storage):
        plot = Asset(
            name="plot",
            storage=storage,
        )

        data = Asset(name="data", storage=storage, serializer="result")

        dataset.add([plot, data])

        assert dataset["plot"] == plot
        assert dataset["data"] == data


class TestAsset:
    @pytest.fixture
    def measurement_results(self, rng):
        results = {f"R{i}": IQResult.random((2048, 50, 2), rng=rng) for i in range(8)}
        return results

    @pytest.fixture
    def dataset_id(self, storage):
        dataset_id = uuid7()
        try:
            yield dataset_id
        finally:
            storage.remove_directory(dataset_id.hex)

    def test_insert_select(self, session, models):
        storage = HTTPStorageBackend.from_url("http://dataserver")
        dataset = Dataset(
            user="qnl",
            sample_id="k210210",
            cooldown_id="SNB230929",
            comments="Test dataset",
        )
        raw_data = Asset(name="raw-data", storage=storage)
        plot = Asset(name="plot", storage=storage)
        dataset.add([raw_data, plot])

        session.add(dataset)
        session.flush()

        results = session.scalars(sa.select(Asset)).all()
        assert len(results) == 2
        assert results == [raw_data, plot]
        assert raw_data.dataset_id == plot.dataset_id == dataset.id

    @pytest.mark.parametrize(
        "asset,fmt",
        [
            (Asset(name="raw-data", serializer="result"), "parquet"),
            (
                Asset(name="raw-data", serializer="result", params=dict(fmt="feather")),
                "feather",
            ),
            (Asset(name="metadata"), "json"),
        ],
    )
    def test_fmt(self, asset, fmt):
        assert asset.fmt == fmt

    @pytest.mark.parametrize(
        "asset,filename",
        [
            (Asset(name="raw-data", serializer="result"), "raw-data.parquet"),
            (
                Asset(
                    name="populations", serializer="result", params=dict(fmt="feather")
                ),
                "populations.feather",
            ),
            (Asset(name="metadata"), "metadata.json"),
        ],
    )
    def test_address(self, asset, filename):
        asset.dataset_id = uuid7()
        assert asset.default_address() == f"/{asset.dataset_id.hex}/{filename}"

    def test_save(self, storage, measurement_results, dataset_id):
        asset = Asset(
            name="raw",
            dataset_id=dataset_id,
            storage=storage,
            serializer="result",
            obj=measurement_results,
        )

        asset.save()

        assert storage.list_directory(asset.dataset_id.hex) == {
            "path": f"/{asset.dataset_id.hex}",
            "num_children": 1,
            "contents": ["raw.parquet"],
        }

    def test_save_no_address(self, storage, measurement_results):
        asset = Asset(
            name="raw", storage=storage, serializer="result", obj=measurement_results
        )

        with pytest.raises(ValueError):
            asset.save()

    def test_roundtrip(self, storage, measurement_results, dataset_id):
        asset = Asset(
            name="raw",
            dataset_id=dataset_id,
            storage=storage,
            serializer="result",
            obj=measurement_results,
        )

        asset.save()
        reloaded = asset.load()
        assert reloaded == measurement_results
