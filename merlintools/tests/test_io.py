import merlintools
import os
from pyxem.signals.electron_diffraction2d import ElectronDiffraction2D

merlin_path = os.path.dirname(merlintools.__file__)


class TestMIB:

    def test_load_mib_with_dm(self):
        dmfilename = os.path.join(merlin_path, "tests",
                                  "test_data", "HAADF.dm3")
        mib_path = os.path.join(merlin_path, "test", "test_data")
        s = merlintools.io.get_merlin_data(mib_path, dmfilename, use_fpd=False, show_progressbar=False)
        assert type(s) is ElectronDiffraction2D
        assert s.axes_manager.signal_shape == (256, 256)
        assert s.axes_manager.navigation_shape == (20, 20)
