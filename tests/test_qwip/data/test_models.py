import sqlalchemy as sa

from qwip.data.models import Asset, Dataset


class TestDataset:
    def test_insert_select(self, session, models):
        dataset = Dataset(
            user="qnl",
            sample_id="K210210",
            cooldown_id="SNB230929",
            comments="Test dataset",
        )

        session.add(dataset)
        session.flush()

        results = session.scalars(sa.select(Dataset)).all()
        assert len(results) == 1
        assert results == [dataset]


class TestAsset:
    def test_insert_select(self, session, models):
        dataset = Dataset(
            user="qnl",
            sample_id="k210210",
            cooldown_id="SNB230929",
            comments="Test dataset",
        )
        raw_data = Asset(name="raw-data", backend="http://dataserver")
        plot = Asset(name="plot", backend="http://dataserver")
        dataset.add([raw_data, plot])

        session.add(dataset)
        session.flush()

        results = session.scalars(sa.select(Asset)).all()
        assert len(results) == 2
        assert results == [raw_data, plot]
        assert raw_data.dataset_id == plot.dataset_id == dataset.id
