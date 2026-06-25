import matplotlib.pyplot as plt
import unittest
import numpy as np
import tempfile
import shutil

import fpd
from merlintools.vendor.fpd import fpd_processing as fpdp
from merlintools.vendor.fpd import fpd_file as fpdf


class TestDPC_Explorer(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tmp_d = tempfile.mkdtemp(prefix="fpd-")

    def test_DPC_Explorer_plot(self):
        print(self.id())
        fpd.DPC_Explorer(64)
        plt.close("all")

    def test_DPC_Explorer_plot_save(self):
        print(self.id())
        b = fpd.DPC_Explorer(64)
        b.save(self.tmp_d)
        plt.close("all")

    def test_DPC_Explorer_ransac(self):
        print(self.id())
        b = fpd.DPC_Explorer(-64, ransac=True, ransac_dict={"residual_threshold": 5.0})
        plt.close("all")

        noise = 10 / 2.0
        self.assertTrue(np.all(b.x <= noise + 2))
        self.assertTrue(np.all(b.y <= noise + 2))

    def test_DPC_Explorer_ransac(self):
        print(self.id())
        b = fpd.DPC_Explorer(-64, yx_range_from_r=True)
        plt.close("all")

        b = fpd.DPC_Explorer(-64, yx_range_from_r=False)
        plt.close("all")

    def test_DPC_Explorer_median(self):
        print(self.id())
        b = fpd.DPC_Explorer(-64, median=True)
        plt.close("all")

        b = fpd.DPC_Explorer(-64, median=True, median_mode=1)
        plt.close("all")

    def test_DPC_Explorer_seq(self):
        print(self.id())
        yx = np.random.rand(2, 2, 64, 64)
        b = fpd.DPC_Explorer(d=yx)
        for i in range(4):
            b._on_seq_next(None)
            b._on_seq_prev(None)
        plt.close("all")

    def test_DPC_Explorer_seq_gets(self):
        print(self.id())
        yx = np.random.rand(3, 2, 64, 64)
        yx -= yx.mean((-2, -1))[..., None, None]

        b = fpd.DPC_Explorer(d=yx)
        b._on_close(None)

        ims = b.get_image_sequence("y")
        ims_multi = b.get_image_sequence(["y", "x"])
        ims_8bit = b.get_image_sequence("y", True)
        ims_multi_8bit = b.get_image_sequence(["y", "x"], True)
        ims_multi_norm = b.get_image_sequence(["rn", "tn"], True)

    def test_DPC_Explorer_seq_gets_non_seq(self):
        print(self.id())
        yx = np.random.rand(2, 64, 64)
        yx -= yx.mean((-2, -1))[..., None, None]

        b = fpd.DPC_Explorer(d=yx)
        b._on_close(None)

        ims = b.get_image_sequence("y")

    def test_DPC_Explorer_pct(self):
        print(self.id())
        fpd.DPC_Explorer(64, pct=1)
        plt.close("all")

        fpd.DPC_Explorer(64, pct=[1, 1])
        plt.close("all")

        fpd.DPC_Explorer(64, pct=[[1, 1], [1, 1]])
        plt.close("all")

    def test_DPC_Explorer_cyx(self):
        print(self.id())
        fpd.DPC_Explorer(64, cyx=(0, 0))
        plt.close("all")

    def test_DPC_Explorer_r_max_r_min__hist_lims(self):
        print(self.id())
        fpd.DPC_Explorer(64, r_min=0.2, r_max=1, hist_lims=[-1, 1, -1, 1])
        plt.close("all")

    def test_DPC_Explorer_cmap(self):
        print(self.id())
        D = fpd.DPC_Explorer(64, cmap="rainbow")
        plt.close("all")

        fpd.DPC_Explorer(64, cmap=plt.cm.rainbow)
        plt.close("all")

        try:
            fpd.DPC_Explorer(64, cmap=np.pi)
            plt.close("all")
        except:
            pass
        else:
            assert False

        D.plot_cmap_ordinates()

    def test_DPC_Explorer_cmap_nfold(self):
        print(self.id())
        D = fpd.DPC_Explorer(64, cmap="rainbow", cmap_nfold=2)
        plt.close("all")

    def test_DPC_Explorer_ransac_flip(self):
        print(self.id())
        fpd.DPC_Explorer(64, ransac=True, flip_y=True, flip_x=True)
        plt.close("all")

    def test_DPC_Explorer_events(self):
        print(self.id())
        D = fpd.DPC_Explorer(64, median=True)
        D._on_YXr(None)
        D._update_gaus(0.5)
        D._update_vectrot(1)
        D._on_cmap("MLP")
        D._on_median_mode("fake_label")
        D._update_cw_plot_theta()
        D._on_sig(None)
        D._on_vec(None)
        D._on_r_min(None)
        D._on_flipy(None)
        D._on_flipx(None)
        D._on_dt(None)
        D._on_ds(None)
        D._on_ransac(None)
        D._on_median(None)
        # D._on_wxy(None)

        plt.close("all")

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.tmp_d)


if __name__ == "__main__":
    unittest.main()
