import sys


# -------------------------------------------------------------
# JUPYTER WIDGET PATCH: Prevent pylops/cupy from breaking event loops
# -------------------------------------------------------------
class HideCuPyFinder:
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("cupy") or fullname.startswith("cupyx"):
            return None  # Tells pylops that CuPy is absolutely not installed
        return None


# Inject the interceptor at the front of Python's import engine
sys.meta_path.insert(0, HideCuPyFinder())
# -------------------------------------------------------------

__version__ = "0.2.6"

__all__ = [
    "fft_tools",
    "fpd_file",
    "fpd_processing",
    "fpd_io",
    "gwy",
    "ransac_tools",
    "segmented_dpc",
    "synthetic_data",
    "tem_tools",
    "mag_tools",
    "utils",
]

# To get sub-modules
for x in __all__:
    exec("from . import %s" % (x))
del x

# Import classes
from .dpc_explorer_class import DPC_Explorer

del dpc_explorer_class

from .segmented_dpc_class import SegmentedDPC

del segmented_dpc_class

from .AlignNR_class import AlignNR

del AlignNR_class
