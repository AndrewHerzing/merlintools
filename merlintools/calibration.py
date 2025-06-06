import numpy as np
import py4DSTEM
import matplotlib.pylab as plt
import h5py
import hyperspy.api as hs
from merlintools.utils import get_lattice_spacings, k_to_mrads
from scipy.optimize import curve_fit
from matplotlib.patches import Wedge
import fpd.fpd_processing as fpdp
import zarr


class RingPatternCalibration:
    def __init__(self, data, material="AuPd"):
        self.data = data
        self.hkls = get_lattice_spacings(material)

    def find_center(self, mask_radius=50, plot_result=False):
        apt = fpdp.virtual_apertures(self.data.shape, (128, 128), (0, mask_radius))
        self.comyx = fpdp.center_of_mass(
            self.data[np.newaxis, np.newaxis, :, :],
            aperture=apt,
            nr=16,
            nc=16,
            print_stats=False,
        )[:, 0, 0]
        if plot_result:
            plt.figure()
            plt.imshow(np.log(self.data + 1), cmap="inferno")
            plt.gca().scatter(self.comyx[1], self.comyx[0])

    def get_qrange(self, radii, ylim=None, xlim=None):
        self.q_range = radii
        fig, ax = plt.subplots(
            1, 2, figsize=(9, 5), gridspec_kw={"width_ratios": [1, 2]}
        )

        ax[0].imshow(np.log(self.data + 1), cmap="gray")
        annulus = Wedge(
            (self.comyx[1], self.comyx[0]),
            radii[1],
            0,
            360,
            width=(radii[1] - radii[0]),
            alpha=0.3,
            color="red",
        )
        ax[0].add_patch(annulus)
        ax[0].axis("off")

        ax[1].scatter(self.radial[0], self.radial[1])
        if ylim is None:
            pass
        else:
            ax[1].set_ylim(ylim[0], ylim[1])
        if xlim is None:
            pass
        else:
            ax[1].set_xlim(xlim[0], xlim[1])
        ax[1].axvline(self.q_range[0], linestyle="--", color="red")
        ax[1].axvline(self.q_range[1], linestyle="--", color="red")
        plt.tight_layout()

    def radial_profile(self, spf=3):
        self.radial = fpdp.radial_profile(self.data, cyx=self.comyx, spf=spf)

    def fit_peak(self, peak_hkl, plot_result=False):
        def _gaussian(x, A, mu, sigma):
            return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        idx = np.where(
            np.logical_and(
                self.radial[0] >= self.q_range[0], self.radial[0] < self.q_range[1]
            )
        )
        _x = self.radial[0][idx]
        _y = self.radial[1][idx]
        _y = self.radial[1][idx] / np.max(self.radial[1][idx])

        p0 = [np.max(_y), _x[np.where(_y == np.max(_y))][0], 1.0]

        popt, pcov = curve_fit(_gaussian, _x, _y, p0=p0)
        self.A, self.mu, self.sigma = (
            np.max(self.radial[1][idx]) * popt[0],
            popt[1],
            popt[2],
        )

        self.calibration = (1 / self.hkls[peak_hkl]) / self.mu
        if plot_result:
            fig, ax = plt.subplots(1)
            ax.scatter(self.radial[0], self.radial[1])
            ax.set_ylim(0, 600)
            ax.set_xlim(0, 140)
            ax.axvline(self.mu, linestyle="--", color="red")

    def check_results(self, ylim=None, xlim=None):
        fig, ax = plt.subplots(1)
        ax.scatter(self.radial[0], self.radial[1])
        if ylim is None:
            pass
        else:
            ax.set_ylim(ylim[0], ylim[1])
        if xlim is None:
            pass
        else:
            ax.set_xlim(xlim[0], xlim[1])

        for i in self.hkls.keys():
            k = 1 / self.hkls[i]
            ax.axvline(k / self.calibration, linestyle="--", color="red")


class CalibrationObject:
    """
    Class to perform calibrations using py4DSTEM.

    Adapted from ACOM_03_Au_NP_sim.ipynb on https://github.com/py4dstem/py4DSTEM_tutorials

    Attributes
    ---
    filename : str
        Data file name
    datacube : py4DSTEM Datacube
        4D STEM data
    coordinates : py4DSTEM Coordinates
        Coordinate structure
    probe_template : NumPy array
        Image of probe for peak finding
    qR : float
        Radius of the transmitted disc in pixels
    qx0, qy0 : float
        Center of transmitted disc in pixels
    """

    def __init__(self):
        self.filename = None
        self.datacube = None
        self.coordinates = None
        self.probe_template = None
        self.qR = None
        self.qx0 = None
        self.qy0 = None

    def load_calibration_data(self, h5file):
        """
        Load a 4D STEM dataset stored via FPD

        Parameters
        ----------
        h5file : str
            Filename to load

        """
        self.filename = h5file
        with h5py.File(h5file, "r") as h5:
            self.datacube = py4DSTEM.io.DataCube(h5["fpd_expt/fpd_data/data"][...])

        self.coordinates = py4DSTEM.io.datastructure.Coordinates(
            self.datacube.R_Nx,
            self.datacube.R_Ny,
            self.datacube.Q_Nx,
            self.datacube.Q_Nx,
            name="coordinates_AuPd",
        )

    def get_probe_size(self, plot_results=True):
        """
        Measure the probe center and size from an image of the transmitted disc.

        Parameters
        ----------
        plot_results : bool
            If True, plot probe image along with virtual bright- and dark-field images

        """
        sum_image = self.datacube.data[:, :, 0:50, 0:50].sum((2, 3))
        [miny, minx] = np.unravel_index(np.argmin(sum_image), sum_image.shape)
        self.probe_template = self.datacube.data[miny, minx, :, :]

        # dp_max = np.max(self.datacube.data, axis=(0, 1))
        self.qR, self.qx0, self.qy0 = py4DSTEM.process.calibration.get_probe_size(
            self.probe_template, thresh_lower=0.1, thresh_upper=0.9
        )

        bf_mask = py4DSTEM.process.virtualimage.make_circ_mask(
            self.datacube, ((self.qx0, self.qy0), self.qR)
        )
        self.BF = (self.datacube.data * bf_mask).sum((2, 3))

        adf_mask = py4DSTEM.process.virtualimage.make_annular_mask(
            self.datacube, ((self.qx0, self.qy0), (5 * self.qR, 10 * self.qR))
        )
        self.ADF = (self.datacube.data * adf_mask).sum((2, 3))

        if plot_results:
            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            ax[0].imshow(np.log(self.probe_template + 1))
            ax[0].set_title("Probe")
            ax[1].imshow(self.BF)
            ax[1].set_title("BF")
            ax[2].imshow(self.ADF)
            ax[2].set_title("ADF")

    def find_origin(self, plot_results=False):
        """
        Detect the center for all scan positions.

        Parameters
        ----------
        plot_results : bool
            If True, plot results of origin determination

        """

        self.qx0_meas, self.qy0_meas = py4DSTEM.process.calibration.get_origin(
            self.datacube
        )
        if plot_results:
            py4DSTEM.visualize.show_image_grid(
                get_ar=lambda i: [self.qx0_meas, self.qy0_meas][i],
                H=1,
                W=2,
                cmap="RdBu",
            )

    def fit_q(self, plot_results=False):
        """
        Fit measured probe center positions to a plane.

        Parameters
        ----------
        plot_results : bool
            If True, plot results of fit

        """
        self.qx0_fit, self.qy0_fit, self.qx0_residuals, self.qy0_residuals = (
            py4DSTEM.process.calibration.fit_origin(
                self.qx0_meas, self.qy0_meas, fitfunction="plane", robust=True
            )
        )
        if plot_results:
            py4DSTEM.visualize.show_image_grid(
                lambda i: [
                    self.qx0_meas,
                    self.qx0_fit,
                    self.qx0_residuals,
                    self.qy0_meas,
                    self.qy0_fit,
                    self.qy0_residuals,
                ][i],
                H=2,
                W=3,
                cmap="RdBu",
            )

        # Store the origin position
        self.coordinates.set_origin(self.qx0_fit, self.qy0_fit)

    def make_probe_template(self, plot_results=False):
        """
        Create a kernel for peak detection from the probe image.

        Parameters
        ----------
        plot_results : bool
            If True, plot the kernel

        """
        self.probe_kernel = (
            py4DSTEM.process.diskdetection.get_probe_kernel_edge_sigmoid(
                self.probe_template,
                self.qR * 0.0,
                self.qR * 4.0,
                origin=(self.qx0, self.qy0),
            )
        )
        if plot_results:
            py4DSTEM.visualize.show_kernel(self.probe_kernel, R=20, L=40, W=2)

    def get_points(self):
        """
        Extract some diffraction patterns to test peak fitting parameters.

        """
        # Select a few DPs on which to test disk detection parameters
        idx = np.unravel_index(np.argsort(self.ADF.flatten())[-5:], self.ADF.shape)

        rxs_cal = idx[1][0:3]
        rys_cal = idx[0][0:3]
        colors = ["r", "c", "g"]

        py4DSTEM.visualize.show_points(
            self.ADF, x=rxs_cal, y=rys_cal, pointcolor=colors, figsize=(8, 8)
        )
        # py4DSTEM.visualize.show_image_grid(get_ar=lambda i:self.datacube.data[rxs_cal[i],rys_cal[i],:,:],
        #                                    H=1,W=3,get_bordercolor=lambda i:colors[i],scaling='log')
        self.rxs = rxs_cal
        self.rys = rys_cal

    def tune_disk_params(
        self,
        corrPower=1.0,
        sigma=1,
        edgeBoundary=4,
        minAbsoluteIntensity=1.0,
        minRelativeIntensity=0.0,
        minPeakSpacing=8,
        maxNumPeaks=100,
        subpixel="poly",
        upsample_factor=16,
    ):
        """
        Optimize disk detection parameters.

        Parameters
        ----------
        Parameters defined in :func:`~py4DSTEM.process.diskdetection.find_Bragg_disks_selected`

        """

        colors = ["r", "c", "g"]
        self.get_points()
        selected_peaks = py4DSTEM.process.diskdetection.find_Bragg_disks_selected(
            datacube=self.datacube,
            probe=self.probe_kernel,
            Rx=self.rxs,
            Ry=self.rys,
            corrPower=corrPower,
            sigma=sigma,
            edgeBoundary=edgeBoundary,
            minAbsoluteIntensity=minAbsoluteIntensity,
            minRelativeIntensity=minRelativeIntensity,
            minPeakSpacing=minPeakSpacing,
            maxNumPeaks=maxNumPeaks,
            subpixel=subpixel,
            upsample_factor=upsample_factor,
        )

        py4DSTEM.visualize.show_image_grid(
            get_ar=lambda i: self.datacube.data[self.rxs[i], self.rys[i], :, :],
            H=1,
            W=3,
            get_bordercolor=lambda i: colors[i],
            get_x=lambda i: selected_peaks[i].data["qx"],
            get_y=lambda i: selected_peaks[i].data["qy"],
            get_pointcolors=lambda i: colors[i],
            scaling="log",
            power=1,
            clipvals="manual",
            min=0,
            max=200,
        )
        self.corrPower = corrPower
        self.sigma = sigma
        self.edgeBoundary = edgeBoundary
        self.minAbsoluteIntensity = minAbsoluteIntensity
        self.minRelativeIntensity = minRelativeIntensity
        self.minPeakSpacing = minPeakSpacing
        self.maxNumPeaks = maxNumPeaks
        self.subpixel = subpixel
        self.upsample_factor = upsample_factor

    def find_disks(self):
        """
        Find disks at all scan positions using py4DSTEM.process.diskdetection.find_Bragg_disks

        """
        self.braggpeaks_raw = py4DSTEM.process.diskdetection.find_Bragg_disks(
            datacube=self.datacube,
            probe=self.probe_kernel,
            corrPower=self.corrPower,
            sigma=self.sigma,
            edgeBoundary=self.edgeBoundary,
            minAbsoluteIntensity=self.minAbsoluteIntensity,
            minRelativeIntensity=self.minRelativeIntensity,
            minPeakSpacing=self.minPeakSpacing,
            maxNumPeaks=self.maxNumPeaks,
            subpixel=self.subpixel,
            upsample_factor=self.upsample_factor,
            name="braggpeaks_cal_raw",
        )

        # Center the disk positions about the origin
        self.braggpeaks_centered = py4DSTEM.process.calibration.center_braggpeaks(
            self.braggpeaks_raw, coords=self.coordinates
        )

    def get_bvm(self):
        """
        Calculate Bragg vector map

        """
        self.bvm_cal = py4DSTEM.process.diskdetection.get_bvm(
            self.braggpeaks_centered, self.datacube.Q_Nx, self.datacube.Q_Ny
        )

        # Plot the Bragg vector map
        py4DSTEM.visualize.show(
            self.bvm_cal, cmap="inferno", clipvals="manual", min=0, max=10
        )

    def get_elliptical_qrange(self, qrange):
        """
        Interactively determine q-range for elliptical correction

        Parameters
        ----------
        qrange : tuple
            Length 2 tuple defining the inner and outter limits of the fit. The
            region is plotted over the Bragg vector map iteratively until optimal

        """
        py4DSTEM.visualize.show(
            self.bvm_cal,
            cmap="gray",
            scaling="power",
            power=0.5,
            clipvals="manual",
            min=0,
            max=100,
            annulus={
                "center": (self.datacube.Q_Nx / 2.0, self.datacube.Q_Ny / 2.0),
                "Ri": qrange[0],
                "Ro": qrange[1],
                "fill": True,
                "color": "y",
                "alpha": 0.5,
            },
        )
        self.qrange = qrange

    def correct_elliptical_distortion(self, plot_result=False):
        """
        Correct elliptical distortion using py4DSTEM.process.calibration.fit_ellipse_1D

        Parameters
        ----------
        plot_results : bool
            If True, plot results of fit

        """
        # Fit the elliptical distortions
        qx0, qy0, a, e, theta = py4DSTEM.process.calibration.fit_ellipse_1D(
            self.bvm_cal,
            (self.datacube.Q_Nx / 2.0, self.datacube.Q_Ny / 2.0),
            self.qrange,
        )

        # Save to Coordinates
        self.coordinates.set_ellipse(a, e, theta)

        # Confirm that elliptical distortions have been removed

        # Correct bragg peak positions, stretching the elliptical semiminor axis to match the semimajor axis length
        braggpeaks_ellipsecorr = (
            py4DSTEM.process.calibration.correct_braggpeak_elliptical_distortions(
                self.braggpeaks_centered,
                (qx0, qy0, a, e, theta),
            )
        )

        # Recompute the bvm
        bvm_ellipsecorr = py4DSTEM.process.diskdetection.get_bragg_vector_map(
            braggpeaks_ellipsecorr, self.datacube.Q_Nx, self.datacube.Q_Ny
        )

        # Fit an ellipse to the elliptically corrected bvm
        qmin = self.qrange[0]
        qmax = self.qrange[1]
        qx0_corr, qy0_corr, a_corr, e_corr, theta_corr = (
            py4DSTEM.process.calibration.fit_ellipse_1D(
                bvm_ellipsecorr, (qx0, qy0), (qmin, qmax)
            )
        )

        if plot_result:
            py4DSTEM.visualize.show_elliptical_fit(
                self.bvm_cal,
                self.qrange,
                (qx0_corr, qy0_corr, a_corr, e_corr, theta_corr),
                cmap="gray",
                scaling="power",
                power=0.5,
                clipvals="manual",
                min=0,
                max=100,
                fill=True,
            )

        # Print the ratio of the semi-axes before and after correction
        print("The ratio of the semiminor to semimajor axes was measured to be")
        print("")
        print("\t{:.2f}% in the original data and".format(100 * e / a))
        print("\t{:.2f}% in the corrected data.".format(100 * e_corr / a_corr))

        self.bvm_ellipsecorr = bvm_ellipsecorr
        self.braggpeaks_ellipsecorr = braggpeaks_ellipsecorr

    def pixel_calibration(self, d_spacing_Ang=1.41, ymax=9e3, dq=0.25):
        """
        Calibrate based on ellipitcally corrected Bragg vector map.

        The Bragg vector map is first radially integrated. Then, peak used for elliptical correction
        is fit with a Gaussian peak. The mean value of the fit is used to calibrate based on the
        known d spacing. the Bragg vector map. The results of the fit are then plotted. A second plot
        is also shown that validates the calibration by indicating the positions of other peaks relative
        to those expected for the d-spacings.

        Parameters
        ----------
        d_spacing_Ang : float
            d-spacing in inverse Angstroms of the ring used for elliptical correction
        ymax : float
            Maximum extent of the vertical axis of the plot.
        dq : float
            Step size to use for the radial integration


        """
        q, I_radial = py4DSTEM.process.utils.radial_integral(
            self.bvm_ellipsecorr, self.datacube.Q_Nx / 2, self.datacube.Q_Ny / 2, dr=dq
        )

        # Fit a gaussian to find a peak location
        qmin, qmax = self.qrange
        A, mu, sigma = py4DSTEM.process.fit.fit_1D_gaussian(q, I_radial, qmin, qmax)
        inv_Ang_per_pixel = 1.0 / (d_spacing_Ang * mu)

        fig, ax = py4DSTEM.visualize.show_qprofile(
            q=q, intensity=I_radial, ymax=ymax, returnfig=True
        )
        ax.vlines((qmin, qmax), 0, ax.get_ylim()[1], color="r")
        ax.vlines(mu, 0, ax.get_ylim()[1], color="g")
        ax.plot(q, py4DSTEM.process.fit.gaussian(q, A, mu, sigma), color="r")

        # Demonstrate consistency with known Au spacings
        spacings_Ang = np.array(list(get_lattice_spacings("AuPd").values()))
        spacings_inv_Ang = 1.0 / spacings_Ang

        fig, ax = py4DSTEM.visualize.show_qprofile(
            q=q * inv_Ang_per_pixel,
            intensity=I_radial,
            ymax=ymax,
            xlabel="q (1/Ang)",
            returnfig=True,
        )
        ax.vlines(spacings_inv_Ang, 0, ax.get_ylim()[1], color="r")
        self.inv_Ang_per_pixel = inv_Ang_per_pixel
        self.q_calibration = q * inv_Ang_per_pixel
        self.I_calibration = I_radial

    def update_calibration(self):
        """
        Store calibration in ACOM coordinates object

        """
        # Store pixel size in coordinates object (ACOM currently hard assumes the units are Å^-1)
        self.coordinates.set_Q_pixel_size(self.inv_Ang_per_pixel)
        self.coordinates.set_Q_pixel_units(r"Å$^{-1}$")

        self.bragg_peaks_calibrated = (
            py4DSTEM.process.calibration.calibrate_Bragg_peaks_pixel_size(
                self.braggpeaks_ellipsecorr, coords=self.coordinates
            )
        )

    def save_results_hspy(self, outfilename):
        """
        Save calibrated Bragg vector map and radial intergration data to Hyperspy signals.

        Parameters
        ----------
        outfilename : str
            Root name for saved files.  Will be appended with '_BVM.hdf5' and
            '_RadialProfile.hdf5'
        """
        bvm = hs.signals.Signal2D(self.bvm_ellipsecorr)
        bvm.axes_manager[0].scale = self.inv_Ang_per_pixel
        bvm.axes_manager[0].units = r"Å$^{-1}$"
        bvm.axes_manager[0].name = "qx"
        bvm.axes_manager[0].offset = -128 * self.inv_Ang_per_pixel
        bvm.axes_manager[1].scale = self.inv_Ang_per_pixel
        bvm.axes_manager[1].units = r"Å$^{-1}$"
        bvm.axes_manager[1].name = "qy"
        bvm.axes_manager[1].offset = -128 * self.inv_Ang_per_pixel
        bvm.save(outfilename + "_BVM.hdf5")

        q = hs.signals.Signal1D(self.I_calibration)
        q.axes_manager[0].scale = self.q_calibration[1] - self.q_calibration[0]
        q.axes_manager[0].offset = self.q_calibration[0]
        q.axes_manager[0].name = "q"
        q.axes_manager[0].units = r"Å$^{-1}$"
        q.save(outfilename + "_RadialProfile.hdf5")

        return

    def save_calibration_results(self, outfilename):
        """
        Save Bragg vector map, radial integration, and calibration value to HDF5 file.

        Parameters
        ----------
        outfilename : float
            File for saving results
        """
        with h5py.File(outfilename, "w") as h5:
            h5.create_dataset("bvm", data=self.bvm_ellipsecorr)
            h5.create_dataset("q", data=self.q_calibration)
            h5.create_dataset("I", data=self.I_calibration)
            h5.create_dataset("calibration", data=self.inv_Ang_per_pixel)

        return


def load_calibration_results(h5filename):
    """
    Load calibration data saved using CalibrationObject.save_calibration_results.

    Parameters
    ----------
    h5filename : float
        File containing calibration results
    """
    cal = {}
    with h5py.File(h5filename) as h5:
        for i in ["bvm", "q", "I", "calibration"]:
            cal[i] = h5[i][...]
    return cal


class CalibrationUpdater:
    def __init__(self, zspy_file):
        self.zspy_file = zspy_file
        self.dp_file = zspy_file[:-5] + "_Sum_DP.hspy"

        s = hs.load(zspy_file, lazy=True)
        self.dp = s.sum((0, 1))
        self.dp.compute()
        self.dp.axes_manager[0].scale = 1.0
        self.dp.axes_manager[1].scale = 1.0
        self.dp.axes_manager[0].offset = -(self.dp.data.shape[1] / 2)
        self.dp.axes_manager[1].offset = -(self.dp.data.shape[0] / 2)

        self.zspy_update = None
        self.beam_energy = self.dp.metadata.Acquisition_instrument.TEM.beam_energy

    def plot_rois(self, vmax=None):
        if vmax is None:
            vmax = 0.01 * self.dp.data.max()
        self.dp.plot(cmap="inferno", vmax=vmax)

        self.roi = hs.roi.CircleROI(cx=-20, cy=-20, r_inner=0, r=10)
        self.roi.add_widget(self.dp)

        self.roi2 = hs.roi.CircleROI(cx=20.0, cy=20.0, r_inner=0, r=10)
        self.roi2.add_widget(self.dp)

    def calculate_calibration(self, dhkl):
        distance = np.sqrt(
            np.square(self.roi.cx - self.roi2.cx)
            + np.square(self.roi.cy - self.roi2.cy)
        )
        g111 = 1 / dhkl
        self.kcal = g111 / (distance / 2)
        print("Calibration: %.3f nm^-1/pixel" % self.kcal)

        alpha_k = self.kcal * (self.roi.r + self.roi2.r) / 2
        self.alpha = k_to_mrads(2 * alpha_k, self.beam_energy)
        self.alpha = round(self.alpha, 1)
        print("Convergence angle: %.1f mrads" % self.alpha)

    def update_kcal(self):
        self.dp.set_diffraction_calibration(self.kcal)
        self.dp.set_experimental_parameters(convergence_angle=self.alpha)
        self.dp.save(self.dp_file, overwrite=True, file_format="HSPY")

        zspy_group = zarr.open(self.zspy_file, "r+")
        zspy_group["Experiments/__unnamed__/axis-2"].attrs["scale"] = self.kcal
        zspy_group["Experiments/__unnamed__/axis-3"].attrs["scale"] = self.kcal
        zspy_group["Experiments/__unnamed__/axis-2"].attrs["offset"] = (
            self.kcal * -128.0
        )
        zspy_group["Experiments/__unnamed__/axis-3"].attrs["offset"] = (
            self.kcal * -128.0
        )
        zspy_group["Experiments/__unnamed__/metadata/Acquisition_instrument"][
            "TEM"
        ].attrs["convergence_angle"] = self.alpha
        zspy_group.store.close()
        return

    def check_kcal(self):
        self.zspy_update = hs.load(self.zspy_file, lazy=True)

        if (
            self.zspy_update.axes_manager[2].scale == self.kcal
            and self.zspy_update.axes_manager[3].scale == self.kcal
        ):
            print("Full DP calibration is correct!")
        else:
            print("Full DP calibration is not correct.")
