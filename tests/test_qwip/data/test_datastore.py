import platform

import pendulum
import pytest

from qwip.data.datastore import Datastore, OfflineDatastore
from qwip.data.models import Asset
from qwip.processing.processors import IQResult


class TestOfflineDatastore:
    @pytest.fixture
    def datastore(self, database, models, storage):
        match database.url.get_backend_name():
            case "sqlite":
                cls = OfflineDatastore
            case "mysql":
                cls = Datastore
            case other:
                raise ValueError(f"Unsupported database dialect: {other}")

        datastore = cls(
            url=database.url,
            storage=storage,
        )

        datastore.engine = database.engine
        datastore.session = database.session
        return datastore

    @pytest.fixture
    def measurement_results(self, rng):
        results = {f"R{i}": IQResult.random((2048, 50, 2), rng=rng) for i in range(8)}
        return results

    def test_save_dataset(self, datastore, session):
        dataset = datastore.save()

        assert dataset.host == platform.node()
        assert dataset.user == datastore.username
        assert dataset.protocol is None
        assert dataset.comments is None
        assert dataset.assets() == ()

    def test_roundtrip_dataset(self, datastore, session):
        dataset = datastore.save(
            sample_id="K230210",
            cooldown_id="SNB230929",
            protocol="T1",
            comments="comment",
        )

        reloaded = datastore.load(dataset.id)
        assert reloaded == dataset

    def test_save_assets(self, datastore, measurement_results, session):
        simple_dict = dict(a=1, b=2, c=3)
        timestamp = pendulum.now()
        result = Asset.create(measurement_results, name="raw", serializer="result")

        dataset = datastore.save(
            metadata=simple_dict,
            time=timestamp,
            raw=result,
        )

        assert dataset["metadata"].load() == simple_dict
        assert dataset["time"].load(cls=pendulum.DateTime) == timestamp
        assert dataset["raw"].load() == measurement_results

        datastore.storage.remove_directory(dataset.id.hex)

    def test_add_asset(self, datastore, measurement_results, session):
        simple_dict = dict(a=1, b=2, c=3)

        dataset = datastore.save(sample_id="K230210", cooldown_id="SNB230929")

        assert dataset.assets() == tuple()

        asset = datastore.add_asset(dataset.id, simple_dict, "metadata")
        assert asset.dataset_id == dataset.id
        assert asset.load() == simple_dict
        print(asset)

        new_dict = dict(a=2, b=3, c=4)
        new_asset = datastore.add_asset(
            dataset.id, new_dict, "metadata", overwrite=True
        )
        assert new_asset is asset
        assert asset.load() == new_dict
