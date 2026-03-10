import numpy as np
import scipy.signal as sig
from scipy.stats import linregress
from typing import Optional, Union, List, Dict, Any, TypeAlias
from src.utils.despike_utils import threshold, goring_nikora, recursive_gaussian
from src.utils.spectral_utils import psd, csd, get_frequency_range
from src.utils.base_instrument import BaseInstrument
from src.utils.constants import (
    GRAVITATIONAL_ACCELERATION as g,
    GAS_CONSTANT_UNIVERSAL as R,
    GAS_CONSTANT_DRY_AIR as R_a,
    GAS_CONSTANT_WATER_VAPOR as R_v,
    MOL_MASS_DRY_AIR as m_a,
    MOL_MASS_WATER_VAPOR as m_v
)

Numeric: TypeAlias = float | int | np.ndarray


class Met(BaseInstrument):
    """
    Class for processing bulk meteorological data. Contains methods for:
    - Loading data from source files
    - Preprocessing
    - Calculating and converting from/to various useful thermodynamic quantities

    Many of the methods are implemented based on their descriptions in Bradley & Fairall (2007). If a particular
    function/equation lacks a citation, it can likely be found in Appendix A therein.

    References
    ----------
    Bradley, E. F., & Fairall, C. W. (2007). A guide to making climate quality meteorological and flux measurements at
        sea.
    """

    def __init__(
        self,
        files: Union[str, List],
        name_map: dict,
        fs: Optional[float] = None,
        z: Optional[Union[float, List[float]]] = None,
        data_keys: Optional[Union[str, List[str]]] = None,
    ):
        """
        Initialize a Met object.

        Parameters
        ----------
        files : str or List[str]
            Path(s) to data files. If a list, each element is treated as a file containing data from
            an individual burst period. Supported formats: .npy (saved as a dict), .mat (saved as a
            MATLAB struct), .csv (variables in columns). If variables are two-dimensional, the larger
            dimension is assumed to be time and the shorter dimension a vertical coordinate.
        name_map : dict
            Mapping of standard variable names to names in the data files, e.g.:
            {
                "t": "temperature variable name" or ["var 1", "var 2", ...],
                "p": "pressure variable name" or ["var 1", "var 2", ...],
                "rh": "relative humidity name" or ["var 1", "var 2", ...],
                "time": "time variable name" or ["var 1", "var 2", ...],
            }
            Lists are used when data from multiple instruments are stored in
            separate variables rather than a 2-D array.
        fs : float, optional
            Sampling frequency (Hz). If not provided, it will be inferred (and rounded to 2 decimal places) from the
            `time` variable
        z : float or List[float], optional
            Mean height above the surface (m) for each instrument. Defaults to integer indices if not
            specified.
        data_keys : str or List[str], optional
            One or more nested keys to traverse after loading the file (e.g. "Data" if the variables
            in name_map are stored at `burst["Data"]["variable_name"]`).

        Returns
        -------
        Met
        """
        files_list = files if isinstance(files, list) else [files]
        Met.validate_inputs(files_list, name_map, fs, z, data_keys)
        super().__init__(files, name_map, fs, z, data_keys)

    @staticmethod
    def validate_inputs(
        files: Union[str, List],
        name_map: dict,
        fs: Optional[Union[int, float]] = None,
        z: Optional[Union[float, int, List[Union[float, int]]]] = None,
        data_keys: Optional[Union[str, List[str]]] = None,
    ):

        # General validation
        BaseInstrument.validate_common_inputs(files, name_map, fs, z, data_keys)


    def set_preprocess_opts(self, opts: Dict[str, Any]):
        """
        Enable preprocessing for all subsequent burst loads using the options defined in the input dictionary.

        Parameters
        ----------
        opts : dict
            Preprocessing options. Supported keys:

            despike : dict, optional
                Options for despiking. If not specified, no despiking is applied. Supported keys:

                method : {'threshold', 'goring_nikora', 'recursive_gaussian'}
                    If `threshold`, data is despiked by replacing any samples with a magnitude outside a specified
                    range. If `goring_nikora`, data is despiked using the Goring & Nikora (2002) algorithm. If
                    `recursive_gaussian`, data is despiked using a recursive Gaussian filter.

                If ``{'method': 'goring_nikora', ...}``, additional keys can be (see `goring_nikora` docstring):
                    remaining_spikes : int
                    max_iter : int
                    robust_statistics : bool

                If ``{'method': 'threshold', ...}``, additional keys can be:
                    threshold_min : float
                    threshold_max : float

                If ``{'method': 'recursive_gaussian', ...}``, additional keys can be:
                    alpha : float
                    max_iter : int
        """

        self._preprocess_opts = opts
        self._preprocess_enabled = True

        self._despike = opts.get("despike", {})
        if self._despike:
            self._despike_method = self._despike.get("method")
            self._despike_opts = {key: val for key, val in self._despike.items() if key != "method"}

        self._cached_idx = None
        self._cached_data = None

    def _apply_preprocessing(self, burst_data):

        if not self._preprocess_enabled:
            return burst_data

        if self._despike:
            despike_fn = {
                "goring_nikora": goring_nikora,
                "threshold": threshold,
                "recursive_guassian": recursive_gaussian,
            }.get(self._despike_method)
            if despike_fn is None:
                raise ValueError(f"Invalid despiking method '{self._despike_method}'")
            for key in self.data_keys:
                burst_data[key] = despike_fn(burst_data[key], **self._despike_opts)

        return burst_data

    def saturation_vapor_pressure(self, p: Numeric, t: Numeric, over_seawater: bool = True) -> Numeric:
        """

        Parameters
        ----------
        p : Numeric
            Atmospheric pressure in millibar
        t : Numeric
            Air temperature in Celcius
        over_seawater : bool, optional
            If true, the saturation vapor pressure is corrected with an assumed salinity 35 PSU

        Returns
        -------
        Saturation vapor pressure in millibar

        """
        e_s = 6.1121 * (1.0007 + 3.46e-6 * p) * np.exp(17.502 * t / (240.97 + t))
        if over_seawater:
            return 0.981 * e_s
        else:
            return e_s

    

    @property
    def data_keys(self):
        return [k for k in self.name_map if k != "time"]