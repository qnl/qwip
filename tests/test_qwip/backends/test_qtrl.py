import numpy as np

from qwip.backends.qtrl import format_legacy_IQ


class TestFormatLegacyIQ:
    def test_reorder(self):
        arr = np.arange(2 * 3 * 4 * 5).astype(float).reshape(2, 3, 4, 5)
        iqdata = format_legacy_IQ(arr)

        assert iqdata.shape == (4, 5, 3)
        assert iqdata.dtype == np.complex128

    def test_float32(self):
        arr = np.arange(2 * 3 * 4 * 5).astype(np.float32).reshape(2, 3, 4, 5)
        iqdata = format_legacy_IQ(arr)

        assert iqdata.shape == (4, 5, 3)
