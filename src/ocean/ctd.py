import numpy as np
from typing import Optional, Union, List, Dict, Any, TypeAlias
from src.utils.despike_utils import threshold, goring_nikora, recursive_gaussian
from src.utils.base_instrument import BaseInstrument
from src.utils.constants import (
    GRAVITATIONAL_ACCELERATION as g,
    SSO,
    T0,
    CP0,
)

Numeric: TypeAlias = float | int | np.ndarray


# =============================================================================
# Private helpers — Conservative Temperature chain
# Direct Python translation of GSW Gibbs-entropy polynomials (TEOS-10).
# Sources: gsw_entropy_part.m, gsw_entropy_part_zerop.m,
#          gsw_gibbs_pt0_pt0.m, gsw_pt0_from_t.m, gsw_CT_from_pt.m
# =============================================================================

def _entropy_part(SA: Numeric, t: Numeric, p: Numeric) -> Numeric:
    """
    Partial specific entropy — SA-independent terms omitted (gsw_entropy_part.m).

    These terms are unnecessary when computing potential temperature from in-situ
    temperature, so they are deliberately excluded for efficiency.
    """
    SA = np.maximum(SA, 0)
    sfac = 0.0248826675584615
    x2 = sfac * SA
    x  = np.sqrt(x2)
    y  = t * 0.025
    z  = p * 1e-4

    g03 = (z * (-270.983805184062 +
           z * (776.153611613101 + z * (-196.51255088122 + (28.9796526294175 - 2.13290083518327 * z) * z))) +
           y * (-24715.571866078 +
           z * (2910.0729080936 + z * (-1513.116771538718 + z * (546.959324647056 + z * (-111.1208127634436 + 8.68841343834394 * z)))) +
           y * (2210.2236124548363 +
           z * (-2017.52334943521 + z * (1498.081172457456 + z * (-718.6359919632359 + (146.4037555781616 - 4.9892131862671505 * z) * z))) +
           y * (-592.743745734632 +
           z * (1591.873781627888 + z * (-1207.261522487504 + (608.785486935364 - 105.4993508931208 * z) * z)) +
           y * (290.12956292128547 +
           z * (-973.091553087975 + z * (602.603274510125 + z * (-276.361526170076 + 32.40953340386105 * z))) +
           y * (-113.90630790850321 + y * (21.35571525415769 - 67.41756835751434 * z) +
           z * (381.06836198507096 + z * (-133.7383902842754 + 49.023632509086724 * z))))))))

    g08 = (x2 * (z * (729.116529735046 +
           z * (-343.956902961561 + z * (124.687671116248 + z * (-31.656964386073 + 7.04658803315449 * z)))) +
           x * (x * (y * (-137.1145018408982 + y * (148.10030845687618 + y * (-68.5590309679152 + 12.4848504784754 * y))) -
           22.6683558512829 * z) +
           z * (-175.292041186547 + (83.1923927801819 - 29.483064349429 * z) * z) +
           y * (-86.1329351956084 +
           z * (766.116132004952 + z * (-108.3834525034224 + 51.2796974779828 * z)) +
           y * (-30.0682112585625 - 1380.9597954037708 * z + y * (3.50240264723578 + 938.26075044542 * z)))) +
           y * (1760.062705994408 +
           y * (-675.802947790203 +
           y * (365.7041791005036 +
           y * (-108.30162043765552 + 12.78101825083098 * y) +
           z * (-1190.914967948748 + (298.904564555024 - 145.9491676006352 * z) * z)) +
           z * (2082.7344423998043 + z * (-614.668925894709 + (340.685093521782 - 33.3848202979239 * z) * z))) +
           z * (-1721.528607567954 + z * (674.819060538734 + z * (-356.629112415276 + (88.4080716616 - 15.84003094423364 * z) * z))))))

    return -(g03 + g08) * 0.025


def _entropy_part_zerop(SA: Numeric, pt0: Numeric) -> Numeric:
    """Partial entropy evaluated at p=0 dbar (gsw_entropy_part_zerop.m)."""
    SA = np.maximum(SA, 0)
    sfac = 0.0248826675584615
    x2 = sfac * SA
    x  = np.sqrt(x2)
    y  = pt0 * 0.025

    g03 = y * (-24715.571866078 +
          y * (2210.2236124548363 +
          y * (-592.743745734632 +
          y * (290.12956292128547 +
          y * (-113.90630790850321 + y * 21.35571525415769)))))

    g08 = (x2 * (x * (x * (y * (-137.1145018408982 +
           y * (148.10030845687618 + y * (-68.5590309679152 + 12.4848504784754 * y)))) +
           y * (-86.1329351956084 + y * (-30.0682112585625 + y * 3.50240264723578))) +
           y * (1760.062705994408 +
           y * (-675.802947790203 +
           y * (365.7041791005036 + y * (-108.30162043765552 + 12.78101825083098 * y))))))

    return -(g03 + g08) * 0.025


def _gibbs_pt0_pt0(SA: Numeric, pt0: Numeric) -> Numeric:
    """Second derivative of Gibbs function w.r.t. temperature at p=0 (gsw_gibbs_pt0_pt0.m)."""
    SA = np.maximum(SA, 0)
    sfac = 0.0248826675584615
    x2 = sfac * SA
    x  = np.sqrt(x2)
    y  = pt0 * 0.025

    g03 = (-24715.571866078 +
           y * (4420.4472249096725 +
           y * (-1778.231237203896 +
           y * (1160.5182516851419 +
           y * (-569.531539542516 + y * 128.13429152494615)))))

    g08 = (x2 * (1760.062705994408 +
           x * (-86.1329351956084 +
           x * (-137.1145018408982 +
           y * (296.20061691375236 + y * (-205.67709290374563 + 49.9394019139016 * y))) +
           y * (-60.136422517125 + y * 10.50720794170734)) +
           y * (-1351.605895580406 +
           y * (1097.1125373015109 +
           y * (-433.20648175062206 + 63.905091254154904 * y)))))

    return (g03 + g08) * 0.000625


def _pt0_from_t(SA: Numeric, t: Numeric, p: Numeric) -> Numeric:
    """
    Potential temperature with p_ref=0 dbar via 2-iteration Newton's method (gsw_pt0_from_t.m).

    Maximum error: 1.8e-14 °C over the full oceanographic funnel.
    """
    SA = np.maximum(SA, 0)
    s1 = SA * (35.0 / SSO)

    pt0 = t + p * (8.65483913395442e-6 -
                   s1 * 1.41636299744881e-6 -
                   p  * 7.38286467135737e-9 +
                   t  * (-8.38241357039698e-6 +
                         s1 * 2.83933368585534e-8 +
                         t  * 1.77803965218656e-8 +
                         p  * 1.71155619208233e-10))

    dentropy_dt = CP0 / ((T0 + pt0) * (1.0 - 0.05 * (1.0 - SA / SSO)))
    true_entropy_part = _entropy_part(SA, t, p)

    for _ in range(2):
        pt0_old = pt0
        dentropy = _entropy_part_zerop(SA, pt0_old) - true_entropy_part
        pt0 = pt0_old - dentropy / dentropy_dt
        pt0m = 0.5 * (pt0 + pt0_old)
        dentropy_dt = -_gibbs_pt0_pt0(SA, pt0m)
        pt0 = pt0_old - dentropy / dentropy_dt

    return pt0


def _CT_from_pt(SA: Numeric, pt: Numeric) -> Numeric:
    """
    Conservative Temperature from potential temperature via potential enthalpy
    polynomial (gsw_CT_from_pt.m).
    """
    SA = np.maximum(SA, 0)
    sfac = 0.0248826675584615
    x2 = sfac * SA
    x  = np.sqrt(x2)
    y  = pt * 0.025

    pot_enthalpy = (61.01362420681071 +
        y * (168776.46138048015 +
        y * (-2735.2785605119625 +
        y * (2574.2164453821433 +
        y * (-1536.6644434977543 +
        y * (545.7340497931629 +
        (-50.91091728474331 - 18.30489878927802 * y) * y))))) +
        x2 * (268.5520265845071 +
        y * (-12019.028203559312 +
        y * (3734.858026725145 +
        y * (-2046.7671145057618 +
        y * (465.28655623826234 +
        (-0.6370820302376359 - 10.650848542359153 * y) * y)))) +
        x * (937.2099110620707 +
        y * (588.1802812170108 +
        y * (248.39476522971285 +
        (-3.871557904936333 - 2.6268019854268356 * y) * y)) +
        x * (-1687.914374187449 +
        x * (246.9598888781377 +
        x * (123.59576582457964 - 48.5891069025409 * x)) +
        y * (936.3206544460336 +
        y * (-942.7827304544439 +
        y * (369.4389437509002 +
        (-33.83664947895248 - 9.987880382780322 * y) * y)))))))

    return 2.505092880681252e-4 * pot_enthalpy  # = pot_enthalpy / CP0


def _eos_vars(SA: Numeric, CT: Numeric, p: Numeric):
    """
    Compute normalised coordinates for the 75-term EOS (Roquet et al., 2015).

    Returns (xs, ys, z) where:
        xs = sqrt(sfac * SA + offset),  sfac = 1/(40*(35.16504/35))
        ys = CT * 0.025
        z  = p  * 1e-4
    """
    sfac   = 0.0248826675584615    # 1 / (40 * (35.16504/35))
    offset = 5.971840214030754e-1  # deltaS * sfac,  deltaS = 24
    xs = np.sqrt(sfac * np.maximum(SA, 0) + offset)
    ys = CT * 0.025
    z  = p  * 1e-4
    return xs, ys, z


# =============================================================================
# CTD class
# =============================================================================

class CTD(BaseInstrument):
    """
    Class for processing CTD (conductivity/temperature/depth) data. Contains
    methods for loading data from source files, preprocessing, and calculating
    thermodynamic quantities from CTD observations.

    The thermodynamic core is a targeted port of the Gibbs SeaWater (GSW)
    Oceanographic Toolbox (TEOS-10, https://www.teos-10.org). Only the
    equation of state and directly derived quantities are implemented; ice
    thermodynamics, gas solubility, and geostrophic functions are omitted.

    Burst dictionary conventions
    ----------------------------
    Variables in a burst dict are assumed to be 2-D arrays of shape
    (n_heights, n_samples), where the first axis corresponds to instrument
    depths (length self.n_heights) and the second axis is time. The individual
    thermodynamic methods accept any Numeric type and broadcast over these
    arrays without modification.

    Standard burst dict keys recognised by :meth:`derive`:

    Input keys
        SP  : Practical Salinity (PSS-78)                         [unitless]
        t   : in-situ temperature                                    [deg C]
        p   : sea pressure (absolute pressure − 10.1325 dbar)         [dbar]
        lat : latitude (scalar)                         [deg N] -- optional

    Output keys added by :meth:`derive`
        SA          : Absolute Salinity                               [g/kg]
        CT          : Conservative Temperature                       [deg C]
        rho         : in-situ density                               [kg/m³]
        sigma0      : potential density anomaly ref 0 dbar           [kg/m³]
        alpha       : thermal expansion coefficient                    [1/K]
        beta        : haline contraction coefficient                  [kg/g]
        sound_speed : speed of sound                                   [m/s]
        t_freezing  : in-situ freezing temperature                   [deg C]
        cp          : isobaric heat capacity                       [J/(kg K)]
        nu          : kinematic viscosity                            [m²/s]
        N2          : buoyancy frequency squared (n_heights > 1)     [1/s²]
        z           : depth (positive downward)                          [m]

    References
    ----------
    IOC, SCOR and IAPSO, 2010: The international thermodynamic equation of
        seawater - 2010. Intergovernmental Oceanographic Commission,
        Manuals and Guides No. 56, UNESCO.
    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
        polynomial expressions for the density and specific volume of seawater
        using the TEOS-10 standard. Ocean Modelling, 90, 29-43.
    McDougall, T.J. and P.M. Barker, 2011: Getting started with TEOS-10 and
        the Gibbs Seawater (GSW) Oceanographic Toolbox. SCOR/IAPSO WG127.
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
        Initialize a CTD object.

        Parameters
        ----------
        files : str or List[str]
            Path(s) to data files. If a list, each element is treated as a file
            containing data from an individual burst period. Supported formats:
            .npy (saved as a dict), .mat (saved as a MATLAB struct), .csv
            (variables in columns). If variables are two-dimensional, the larger
            dimension is assumed to be time and the shorter dimension a vertical
            coordinate.
        name_map : dict
            Mapping of standard variable names to names in the data files, e.g.:
            {
                "SP": "salinity variable name",
                "t":  "temperature variable name",
                "p":  "pressure variable name",
                "time": "time variable name",
            }
            Lists are used when data from multiple instruments are stored in
            separate variables rather than a 2-D array.
        fs : float, optional
            Sampling frequency (Hz). If not provided, it will be inferred (and
            rounded to 2 decimal places) from the ``time`` variable.
        z : float or List[float], optional
            Instrument depth(s) below the surface (m). Defaults to integer
            indices if not specified.
        data_keys : str or List[str], optional
            One or more nested keys to traverse after loading the file (e.g.
            ``"Data"`` if the variables in name_map are stored at
            ``burst["Data"]["variable_name"]``).

        Returns
        -------
        CTD
        """
        files_list = files if isinstance(files, list) else [files]
        CTD.validate_inputs(files_list, name_map, fs, z, data_keys)
        super().__init__(files, name_map, fs, z, data_keys)

    @staticmethod
    def validate_inputs(
        files: Union[str, List],
        name_map: dict,
        fs: Optional[Union[int, float]] = None,
        z: Optional[Union[float, int, List[Union[float, int]]]] = None,
        data_keys: Optional[Union[str, List[str]]] = None,
    ):
        BaseInstrument.validate_common_inputs(files, name_map, fs, z, data_keys)

    def set_preprocess_opts(self, opts: Dict[str, Any]):
        """
        Enable preprocessing for all subsequent burst loads.

        Parameters
        ----------
        opts : dict
            Preprocessing options. Supported keys:

            despike : dict, optional
                Options for despiking. Supported keys:

                method : {'threshold', 'goring_nikora', 'recursive_gaussian'}
                    Despiking algorithm to apply.

                If ``{'method': 'goring_nikora', ...}``, additional keys:
                    remaining_spikes : int
                    max_iter : int
                    robust_statistics : bool

                If ``{'method': 'threshold', ...}``, additional keys:
                    threshold_min : float
                    threshold_max : float

                If ``{'method': 'recursive_gaussian', ...}``, additional keys:
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
                "recursive_gaussian": recursive_gaussian,
            }.get(self._despike_method)
            if despike_fn is None:
                raise ValueError(f"Invalid despiking method '{self._despike_method}'")
            var_keys = [k for k in self.name_map if k != "time"]
            for key in var_keys:
                burst_data[key] = despike_fn(burst_data[key], **self._despike_opts)

        return burst_data

    # -------------------------------------------------------------------------
    # Salinity and temperature conversions
    # -------------------------------------------------------------------------

    def SA_from_SP(self, SP: Numeric) -> Numeric:
        """
        Absolute Salinity from Practical Salinity using the constant-ratio
        approximation (gsw_SA_from_SP.m, simplified).

        Uses SA = SP × (35.16504 / 35), which skips the geographic Absolute
        Salinity Anomaly (SAAR) correction. Typical error is < 0.025 g/kg in
        the open ocean. Errors can reach ~0.1 g/kg in marginal seas (Baltic,
        Red Sea, Arctic shelf) where SAAR is significant. For deployments
        anywhere in the world ocean this approximation is the accepted trade-off
        when geographic position is unavailable.

        Parameters
        ----------
        SP : Numeric
            Practical Salinity (PSS-78) [unitless]

        Returns
        -------
        Numeric
            Absolute Salinity [g/kg]
        """
        return np.maximum(SP, 0) * (SSO / 35.0)

    def CT_from_t(self, SA: Numeric, t: Numeric, p: Numeric) -> Numeric:
        """
        Conservative Temperature from in-situ temperature (gsw_CT_from_t.m).

        Computes potential temperature at p_ref = 0 dbar via two iterations of
        Newton's method using Gibbs-entropy polynomials, then converts to
        Conservative Temperature via the potential-enthalpy polynomial.

        Parameters
        ----------
        SA : Numeric
            Absolute Salinity [g/kg]
        t : Numeric
            In-situ temperature (ITS-90) [deg C]
        p : Numeric
            Sea pressure [dbar]

        Returns
        -------
        Numeric
            Conservative Temperature (ITS-90) [deg C]
        """
        pt0 = _pt0_from_t(SA, t, p)
        return _CT_from_pt(SA, pt0)

    # -------------------------------------------------------------------------
    # 75-term equation of state (Roquet et al., 2015)
    # Coefficients and polynomial structure match gsw_specvol.m / gsw_rho.m /
    # gsw_alpha.m / gsw_beta.m / gsw_sound_speed.m / gsw_sigma0.m exactly.
    # -------------------------------------------------------------------------

    def specific_volume(self, SA: Numeric, CT: Numeric, p: Numeric) -> Numeric:
        """
        Specific volume from the 75-term polynomial EOS (gsw_specvol.m).

        Parameters
        ----------
        SA : Numeric
            Absolute Salinity [g/kg]
        CT : Numeric
            Conservative Temperature [deg C]
        p : Numeric
            Sea pressure [dbar]

        Returns
        -------
        Numeric
            Specific volume [m³/kg]
        """
        xs, ys, z = _eos_vars(SA, CT, p)

        v000 =  1.0769995862e-3
        v001 = -6.0799143809e-5
        v002 =  9.9856169219e-6
        v003 = -1.1309361437e-6
        v004 =  1.0531153080e-7
        v005 = -1.2647261286e-8
        v006 =  1.9613503930e-9
        v010 = -1.5649734675e-5
        v011 =  1.8505765429e-5
        v012 = -1.1736386731e-6
        v013 = -3.6527006553e-7
        v014 =  3.1454099902e-7
        v020 =  2.7762106484e-5
        v021 = -1.1716606853e-5
        v022 =  2.1305028740e-6
        v023 =  2.8695905159e-7
        v030 = -1.6521159259e-5
        v031 =  7.9279656173e-6
        v032 = -4.6132540037e-7
        v040 =  6.9111322702e-6
        v041 = -3.4102187482e-6
        v042 = -6.3352916514e-8
        v050 = -8.0539615540e-7
        v051 =  5.0736766814e-7
        v060 =  2.0543094268e-7
        v100 = -3.1038981976e-4
        v101 =  2.4262468747e-5
        v102 = -5.8484432984e-7
        v103 =  3.6310188515e-7
        v104 = -1.1147125423e-7
        v110 =  3.5009599764e-5
        v111 = -9.5677088156e-6
        v112 = -5.5699154557e-6
        v113 = -2.7295696237e-7
        v120 = -3.7435842344e-5
        v121 = -2.3678308361e-7
        v122 =  3.9137387080e-7
        v130 =  2.4141479483e-5
        v131 = -3.4558773655e-6
        v132 =  7.7618888092e-9
        v140 = -8.7595873154e-6
        v141 =  1.2956717783e-6
        v150 = -3.3052758900e-7
        v200 =  6.6928067038e-4
        v201 = -3.4792460974e-5
        v202 = -4.8122251597e-6
        v203 =  1.6746303780e-8
        v210 = -4.3592678561e-5
        v211 =  1.1100834765e-5
        v212 =  5.4620748834e-6
        v220 =  3.5907822760e-5
        v221 =  2.9283346295e-6
        v222 = -6.5731104067e-7
        v230 = -1.4353633048e-5
        v231 =  3.1655306078e-7
        v240 =  4.3703680598e-6
        v300 = -8.5047933937e-4
        v301 =  3.7470777305e-5
        v302 =  4.9263106998e-6
        v310 =  3.4532461828e-5
        v311 = -9.8447117844e-6
        v312 = -1.3544185627e-6
        v320 = -1.8698584187e-5
        v321 = -4.8826139200e-7
        v330 =  2.2863324556e-6
        v400 =  5.8086069943e-4
        v401 = -1.7322218612e-5
        v402 = -1.7811974727e-6
        v410 = -1.1959409788e-5
        v411 =  2.5909225260e-6
        v420 =  3.8595339244e-6
        v500 = -2.1092370507e-4
        v501 =  3.0927427253e-6
        v510 =  1.3864594581e-6
        v600 =  3.1932457305e-5

        v = (v000 + xs*(v100 + xs*(v200 + xs*(v300 + xs*(v400 + xs*(v500
            + v600*xs))))) + ys*(v010 + xs*(v110 + xs*(v210 + xs*(v310 + xs*(v410
            + v510*xs)))) + ys*(v020 + xs*(v120 + xs*(v220 + xs*(v320 + v420*xs)))
            + ys*(v030 + xs*(v130 + xs*(v230 + v330*xs)) + ys*(v040 + xs*(v140
            + v240*xs) + ys*(v050 + v150*xs + v060*ys))))) + z*(v001 + xs*(v101
            + xs*(v201 + xs*(v301 + xs*(v401 + v501*xs)))) + ys*(v011 + xs*(v111
            + xs*(v211 + xs*(v311 + v411*xs))) + ys*(v021 + xs*(v121 + xs*(v221
            + v321*xs)) + ys*(v031 + xs*(v131 + v231*xs) + ys*(v041 + v141*xs
            + v051*ys)))) + z*(v002 + xs*(v102 + xs*(v202 + xs*(v302 + v402*xs)))
            + ys*(v012 + xs*(v112 + xs*(v212 + v312*xs)) + ys*(v022 + xs*(v122
            + v222*xs) + ys*(v032 + v132*xs + v042*ys))) + z*(v003 + xs*(v103
            + v203*xs) + ys*(v013 + v113*xs + v023*ys) + z*(v004 + v104*xs + v014*ys
            + z*(v005 + v006*z))))))

        return v

    def density(self, SA: Numeric, CT: Numeric, p: Numeric) -> Numeric:
        """
        In-situ density from the 75-term polynomial EOS (gsw_rho.m).

        Parameters
        ----------
        SA : Numeric
            Absolute Salinity [g/kg]
        CT : Numeric
            Conservative Temperature [deg C]
        p : Numeric
            Sea pressure [dbar]

        Returns
        -------
        Numeric
            In-situ density [kg/m³]
        """
        return 1.0 / self.specific_volume(SA, CT, p)

    def alpha(self, SA: Numeric, CT: Numeric, p: Numeric) -> Numeric:
        """
        Thermal expansion coefficient with respect to Conservative Temperature
        from the 75-term polynomial EOS (gsw_alpha.m).

        Parameters
        ----------
        SA : Numeric
            Absolute Salinity [g/kg]
        CT : Numeric
            Conservative Temperature [deg C]
        p : Numeric
            Sea pressure [dbar]

        Returns
        -------
        Numeric
            Thermal expansion coefficient [1/K]
        """
        xs, ys, z = _eos_vars(SA, CT, p)

        # v_CT_part = dv/d(ys); a_ijk coefficients from gsw_alpha.m
        a000 = -1.5649734675e-5
        a001 =  1.8505765429e-5
        a002 = -1.1736386731e-6
        a003 = -3.6527006553e-7
        a004 =  3.1454099902e-7
        a010 =  5.5524212968e-5
        a011 = -2.3433213706e-5
        a012 =  4.2610057480e-6
        a013 =  5.7391810318e-7
        a020 = -4.9563477777e-5
        a021 =  2.37838968519e-5
        a022 = -1.38397620111e-6
        a030 =  2.76445290808e-5
        a031 = -1.36408749928e-5
        a032 = -2.53411666056e-7
        a040 = -4.0269807770e-6
        a041 =  2.5368383407e-6
        a050 =  1.23258565608e-6
        a100 =  3.5009599764e-5
        a101 = -9.5677088156e-6
        a102 = -5.5699154557e-6
        a103 = -2.7295696237e-7
        a110 = -7.4871684688e-5
        a111 = -4.7356616722e-7
        a112 =  7.8274774160e-7
        a120 =  7.2424438449e-5
        a121 = -1.03676320965e-5
        a122 =  2.32856664276e-8
        a130 = -3.50383492616e-5
        a131 =  5.1826871132e-6
        a140 = -1.6526379450e-6
        a200 = -4.3592678561e-5
        a201 =  1.1100834765e-5
        a202 =  5.4620748834e-6
        a210 =  7.1815645520e-5
        a211 =  5.8566692590e-6
        a212 = -1.31462208134e-6
        a220 = -4.3060899144e-5
        a221 =  9.4965918234e-7
        a230 =  1.74814722392e-5
        a300 =  3.4532461828e-5
        a301 = -9.8447117844e-6
        a302 = -1.3544185627e-6
        a310 = -3.7397168374e-5
        a311 = -9.7652278400e-7
        a320 =  6.8589973668e-6
        a400 = -1.1959409788e-5
        a401 =  2.5909225260e-6
        a410 =  7.7190678488e-6
        a500 =  1.3864594581e-6

        v_CT_part = (a000 + xs*(a100 + xs*(a200 + xs*(a300 + xs*(a400 + a500*xs))))
            + ys*(a010 + xs*(a110 + xs*(a210 + xs*(a310 + a410*xs)))
            + ys*(a020 + xs*(a120 + xs*(a220 + a320*xs))
            + ys*(a030 + xs*(a130 + a230*xs)
            + ys*(a040 + a140*xs + a050*ys))))
            + z*(a001 + xs*(a101 + xs*(a201 + xs*(a301 + a401*xs)))
            + ys*(a011 + xs*(a111 + xs*(a211 + a311*xs))
            + ys*(a021 + xs*(a121 + a221*xs)
            + ys*(a031 + a131*xs + a041*ys)))
            + z*(a002 + xs*(a102 + xs*(a202 + a302*xs))
            + ys*(a012 + xs*(a112 + a212*xs)
            + ys*(a022 + a122*xs + a032*ys))
            + z*(a003 + a103*xs + a013*ys + a004*z))))

        return 0.025 * v_CT_part / self.specific_volume(SA, CT, p)

    def beta(self, SA: Numeric, CT: Numeric, p: Numeric) -> Numeric:
        """
        Haline contraction coefficient at constant Conservative Temperature
        from the 75-term polynomial EOS (gsw_beta.m).

        Parameters
        ----------
        SA : Numeric
            Absolute Salinity [g/kg]
        CT : Numeric
            Conservative Temperature [deg C]
        p : Numeric
            Sea pressure [dbar]

        Returns
        -------
        Numeric
            Haline contraction coefficient [kg/g]
        """
        xs, ys, z = _eos_vars(SA, CT, p)
        sfac = 0.0248826675584615

        # v_SA_part = dv/d(xs) * (xs / SA derivative chain); b_ijk from gsw_beta.m
        b000 = -3.1038981976e-4
        b001 =  2.4262468747e-5
        b002 = -5.8484432984e-7
        b003 =  3.6310188515e-7
        b004 = -1.1147125423e-7
        b010 =  3.5009599764e-5
        b011 = -9.5677088156e-6
        b012 = -5.5699154557e-6
        b013 = -2.7295696237e-7
        b020 = -3.7435842344e-5
        b021 = -2.3678308361e-7
        b022 =  3.9137387080e-7
        b030 =  2.4141479483e-5
        b031 = -3.4558773655e-6
        b032 =  7.7618888092e-9
        b040 = -8.7595873154e-6
        b041 =  1.2956717783e-6
        b050 = -3.3052758900e-7
        b100 =  1.33856134076e-3
        b101 = -6.9584921948e-5
        b102 = -9.62445031940e-6
        b103 =  3.3492607560e-8
        b110 = -8.7185357122e-5
        b111 =  2.2201669530e-5
        b112 =  1.09241497668e-5
        b120 =  7.1815645520e-5
        b121 =  5.8566692590e-6
        b122 = -1.31462208134e-6
        b130 = -2.8707266096e-5
        b131 =  6.3310612156e-7
        b140 =  8.7407361196e-6
        b200 = -2.55143801811e-3
        b201 =  1.12412331915e-4
        b202 =  1.47789320994e-5
        b210 =  1.03597385484e-4
        b211 = -2.95341353532e-5
        b212 = -4.0632556881e-6
        b220 = -5.6095752561e-5
        b221 = -1.4647841760e-6
        b230 =  6.8589973668e-6
        b300 =  2.32344279772e-3
        b301 = -6.9288874448e-5
        b302 = -7.1247898908e-6
        b310 = -4.7837639152e-5
        b311 =  1.0363690104e-5
        b320 =  1.54381356976e-5
        b400 = -1.05461852535e-3
        b401 =  1.54637136265e-5
        b410 =  6.9322972905e-6
        b500 =  1.9159474383e-4

        v_SA_part = (b000 + xs*(b100 + xs*(b200 + xs*(b300 + xs*(b400 + b500*xs))))
            + ys*(b010 + xs*(b110 + xs*(b210 + xs*(b310 + b410*xs)))
            + ys*(b020 + xs*(b120 + xs*(b220 + b320*xs))
            + ys*(b030 + xs*(b130 + b230*xs)
            + ys*(b040 + b140*xs + b050*ys))))
            + z*(b001 + xs*(b101 + xs*(b201 + xs*(b301 + b401*xs)))
            + ys*(b011 + xs*(b111 + xs*(b211 + b311*xs))
            + ys*(b021 + xs*(b121 + b221*xs)
            + ys*(b031 + b131*xs + b041*ys)))
            + z*(b002 + xs*(b102 + xs*(b202 + b302*xs))
            + ys*(b012 + xs*(b112 + b212*xs)
            + ys*(b022 + b122*xs + b032*ys))
            + z*(b003 + b103*xs + b013*ys + b004*z))))

        return -v_SA_part * 0.5 * sfac / (self.specific_volume(SA, CT, p) * xs)

    def sound_speed(self, SA: Numeric, CT: Numeric, p: Numeric) -> Numeric:
        """
        Speed of sound in seawater from the 75-term polynomial EOS (gsw_sound_speed.m).

        Parameters
        ----------
        SA : Numeric
            Absolute Salinity [g/kg]
        CT : Numeric
            Conservative Temperature [deg C]
        p : Numeric
            Sea pressure [dbar]

        Returns
        -------
        Numeric
            Speed of sound [m/s]
        """
        xs, ys, z = _eos_vars(SA, CT, p)

        # v_p = dv/d(z) * 1e-4; c_ijk coefficients from gsw_sound_speed.m
        c000 = -6.0799143809e-5
        c001 =  1.99712338438e-5
        c002 = -3.3928084311e-6
        c003 =  4.2124612320e-7
        c004 = -6.3236306430e-8
        c005 =  1.1768102358e-8
        c010 =  1.8505765429e-5
        c011 = -2.3472773462e-6
        c012 = -1.09581019659e-6
        c013 =  1.25816399608e-6
        c020 = -1.1716606853e-5
        c021 =  4.2610057480e-6
        c022 =  8.6087715477e-7
        c030 =  7.9279656173e-6
        c031 = -9.2265080074e-7
        c040 = -3.4102187482e-6
        c041 = -1.26705833028e-7
        c050 =  5.0736766814e-7
        c100 =  2.4262468747e-5
        c101 = -1.16968865968e-6
        c102 =  1.08930565545e-6
        c103 = -4.4588501692e-7
        c110 = -9.5677088156e-6
        c111 = -1.11398309114e-5
        c112 = -8.1887088711e-7
        c120 = -2.3678308361e-7
        c121 =  7.8274774160e-7
        c130 = -3.4558773655e-6
        c131 =  1.55237776184e-8
        c140 =  1.2956717783e-6
        c200 = -3.4792460974e-5
        c201 = -9.6244503194e-6
        c202 =  5.0238911340e-8
        c210 =  1.1100834765e-5
        c211 =  1.09241497668e-5
        c220 =  2.9283346295e-6
        c221 = -1.31462208134e-6
        c230 =  3.1655306078e-7
        c300 =  3.7470777305e-5
        c301 =  9.8526213996e-6
        c310 = -9.8447117844e-6
        c311 = -2.7088371254e-6
        c320 = -4.8826139200e-7
        c400 = -1.7322218612e-5
        c401 = -3.5623949454e-6
        c410 =  2.5909225260e-6
        c500 =  3.0927427253e-6

        v_p = (c000 + xs*(c100 + xs*(c200 + xs*(c300 + xs*(c400 + c500*xs))))
            + ys*(c010 + xs*(c110 + xs*(c210 + xs*(c310 + c410*xs)))
            + ys*(c020 + xs*(c120 + xs*(c220 + c320*xs))
            + ys*(c030 + xs*(c130 + c230*xs)
            + ys*(c040 + c140*xs + c050*ys))))
            + z*(c001 + xs*(c101 + xs*(c201 + xs*(c301 + c401*xs)))
            + ys*(c011 + xs*(c111 + xs*(c211 + c311*xs))
            + ys*(c021 + xs*(c121 + c221*xs)
            + ys*(c031 + c131*xs + c041*ys)))
            + z*(c002 + xs*(c102 + c202*xs)
            + ys*(c012 + c112*xs + c022*ys)
            + z*(c003 + c103*xs + c013*ys + z*(c004 + c005*z)))))

        v = self.specific_volume(SA, CT, p)
        return 10000.0 * np.sqrt(-v ** 2 / v_p)

    def sigma0(self, SA: Numeric, CT: Numeric) -> Numeric:
        """
        Potential density anomaly referenced to 0 dbar from the 75-term EOS
        (gsw_sigma0.m). Equal to potential density minus 1000 kg/m³.

        Parameters
        ----------
        SA : Numeric
            Absolute Salinity [g/kg]
        CT : Numeric
            Conservative Temperature [deg C]

        Returns
        -------
        Numeric
            Potential density anomaly [kg/m³]
        """
        sfac   = 0.0248826675584615
        offset = 5.971840214030754e-1
        xs = np.sqrt(sfac * np.maximum(SA, 0) + offset)
        ys = CT * 0.025

        # 75-term polynomial evaluated at p=0 (z=0); all z-dependent terms drop out.
        # Only the non-commented coefficients from gsw_sigma0.m are needed.
        v000 =  1.0769995862e-3
        v010 = -1.5649734675e-5
        v020 =  2.7762106484e-5
        v030 = -1.6521159259e-5
        v040 =  6.9111322702e-6
        v050 = -8.0539615540e-7
        v060 =  2.0543094268e-7
        v100 = -3.1038981976e-4
        v110 =  3.5009599764e-5
        v120 = -3.7435842344e-5
        v130 =  2.4141479483e-5
        v140 = -8.7595873154e-6
        v150 = -3.3052758900e-7
        v200 =  6.6928067038e-4
        v210 = -4.3592678561e-5
        v220 =  3.5907822760e-5
        v230 = -1.4353633048e-5
        v240 =  4.3703680598e-6
        v300 = -8.5047933937e-4
        v310 =  3.4532461828e-5
        v320 = -1.8698584187e-5
        v330 =  2.2863324556e-6
        v400 =  5.8086069943e-4
        v410 = -1.1959409788e-5
        v420 =  3.8595339244e-6
        v500 = -2.1092370507e-4
        v510 =  1.3864594581e-6
        v600 =  3.1932457305e-5

        vp0 = (v000 + xs*(v100 + xs*(v200 + xs*(v300 + xs*(v400 + xs*(v500
             + v600*xs))))) + ys*(v010 + xs*(v110 + xs*(v210 + xs*(v310 + xs*(v410
             + v510*xs)))) + ys*(v020 + xs*(v120 + xs*(v220 + xs*(v320 + v420*xs)))
             + ys*(v030 + xs*(v130 + xs*(v230 + v330*xs)) + ys*(v040 + xs*(v140
             + v240*xs) + ys*(v050 + v150*xs + v060*ys))))))

        return 1.0 / vp0 - 1000.0

    # -------------------------------------------------------------------------
    # Derived quantities
    # -------------------------------------------------------------------------

    def freezing_temperature(self, SA: Numeric, p: Numeric) -> Numeric:
        """
        In-situ freezing temperature from a direct polynomial fit (gsw_t_freezing_poly.m).

        Uses the 23-coefficient polynomial given in the comments of gsw_t_freezing_poly.m,
        which avoids calling CT_freezing and t_from_CT. Error is between -8e-4 K and
        +3e-4 K compared with the exact Newton-Raphson method.

        Parameters
        ----------
        SA : Numeric
            Absolute Salinity [g/kg]
        p : Numeric
            Sea pressure [dbar]

        Returns
        -------
        Numeric
            Freezing temperature [deg C]
        """
        SA_r = np.maximum(SA, 0) * 1e-2
        x    = np.sqrt(SA_r)
        p_r  = p * 1e-4

        c0  =  0.002519
        c1  = -5.946302841607319
        c2  =  4.136051661346983
        c3  = -1.115150523403847e1
        c4  =  1.476878746184548e1
        c5  = -1.088873263630961e1
        c6  =  2.961018839640730
        c7  = -7.433320943962606
        c8  = -1.561578562479883
        c9  =  4.073774363480365e-2
        c10 =  1.158414435887717e-2
        c11 = -4.122639292422863e-1
        c12 = -1.123186915628260e-1
        c13 =  5.715012685553502e-1
        c14 =  2.021682115652684e-1
        c15 =  4.140574258089767e-2
        c16 = -6.034228641903586e-1
        c17 = -1.205825928146808e-2
        c18 = -2.812172968619369e-1
        c19 =  1.877244474023750e-2
        c20 = -1.204395563789007e-1
        c21 =  2.349147739749606e-1
        c22 =  2.748444541144219e-3

        return (c0
            + SA_r * (c1 + x*(c2 + x*(c3 + x*(c4 + x*(c5 + c6*x)))))
            + p_r  * (c7 + p_r*(c8 + c9*p_r))
            + SA_r * p_r * (c10 + p_r*(c12 + p_r*(c15 + c21*SA_r))
                          + SA_r*(c13 + c17*p_r + c19*SA_r)
                          + x*(c11 + p_r*(c14 + c18*p_r)
                               + SA_r*(c16 + c20*p_r + c22*SA_r))))

    def heat_capacity(self, SA: Numeric, t: Numeric, p: Numeric) -> Numeric:
        """
        Isobaric specific heat capacity of seawater.

        Uses the UNESCO (1983) empirical polynomial based on Millero et al. (1973)
        with a pressure correction from Fofonoff (1985).

        Parameters
        ----------
        SA : Numeric
            Absolute Salinity [g/kg]
        t : Numeric
            In-situ temperature [deg C]
        p : Numeric
            Sea pressure [dbar]

        Returns
        -------
        Numeric
            Isobaric heat capacity [J/(kg K)]
        """
        # Heat capacity of pure water at 0 dbar (Millero et al., 1973)
        cp_w = (4217.4
                + t * (-3.720283
                + t * ( 0.1412855
                + t * (-2.654387e-3 + 2.093236e-5 * t))))

        # Salinity corrections
        A = -7.644 + t * (0.107276 - 1.3839e-3 * t)
        B = 0.17709
        cp_0 = cp_w + SA * A + SA ** 1.5 * B

        # Pressure correction [J/(kg K) per dbar] (Fofonoff, 1985)
        dp = p * (-5.035e-4 + t * (1.027e-5 - 8.13e-8 * t) + SA * 1.67e-6)

        return cp_0 + dp

    def dynamic_viscosity(self, t: Numeric, SA: Numeric) -> Numeric:
        """
        Dynamic viscosity of seawater (Sharqawy et al., 2010).

        Parameters
        ----------
        t : Numeric
            In-situ temperature [deg C]
        SA : Numeric
            Absolute Salinity [g/kg]

        Returns
        -------
        Numeric
            Dynamic viscosity [Pa s]
        """
        mu_w = 4.2844e-5 + 1.0 / (0.157 * (t + 64.993) ** 2 - 91.296)
        S_kg = SA * 1e-3  # g/kg → kg/kg
        A = 1.541 + 0.0022 * t - 1.06e-5 * t ** 2
        B = 0.333 + 0.002 * t
        return mu_w * (1.0 + A * S_kg + B * S_kg ** 2)

    def kinematic_viscosity(self, t: Numeric, SA: Numeric) -> Numeric:
        """
        Kinematic viscosity of seawater.

        Parameters
        ----------
        t : Numeric
            In-situ temperature [deg C]
        SA : Numeric
            Absolute Salinity [g/kg]

        Returns
        -------
        Numeric
            Kinematic viscosity [m²/s]
        """
        CT  = self.CT_from_t(SA, t, np.zeros_like(t))
        rho = self.density(SA, CT, np.zeros_like(t))
        return self.dynamic_viscosity(t, SA) / rho

    def thermal_conductivity(self, t: Numeric, SA: Numeric) -> Numeric:
        """
        Thermal conductivity of seawater (Caldwell, 1974).

        Parameters
        ----------
        t : Numeric
            In-situ temperature [deg C]
        SA : Numeric
            Absolute Salinity [g/kg]

        Returns
        -------
        Numeric
            Thermal conductivity [W/(m K)]
        """
        return 0.57109 + 1.7499e-3 * t - 6.993e-6 * t ** 2 + 1.5e-3 * (SA / SSO)

    def buoyancy_frequency(
        self,
        SA: np.ndarray,
        CT: np.ndarray,
        p: np.ndarray,
    ) -> np.ndarray:
        """
        Squared buoyancy (Brunt-Väisälä) frequency from a vertical profile.

        Computed via finite differences of Conservative Temperature and Absolute
        Salinity using the thermal expansion and haline contraction coefficients
        from the 75-term EOS:

            N² = g × (α × dCT/dz − β × dSA/dz)

        where z is positive upward (taken from self.z). N² is evaluated at
        mid-points between adjacent instrument depths, so the output has length
        n_heights − 1 along axis 0.

        Requires n_heights > 1 (i.e., the mooring must have sensors at more than
        one depth).

        Parameters
        ----------
        SA : np.ndarray
            Absolute Salinity, shape (n_heights, n_samples) [g/kg]
        CT : np.ndarray
            Conservative Temperature, shape (n_heights, n_samples) [deg C]
        p : np.ndarray
            Sea pressure, shape (n_heights, n_samples) [dbar]

        Returns
        -------
        np.ndarray
            N² at mid-depth levels, shape (n_heights − 1, n_samples) [1/s²]
        """
        SA_mid = 0.5 * (SA[:-1] + SA[1:])
        CT_mid = 0.5 * (CT[:-1] + CT[1:])
        p_mid  = 0.5 * (p[:-1]  + p[1:])

        alpha_mid = self.alpha(SA_mid, CT_mid, p_mid)
        beta_mid  = self.beta(SA_mid, CT_mid, p_mid)

        dz  = np.diff(self.z)                           # shape (n_heights-1,)
        dCT = np.diff(CT, axis=0)
        dSA = np.diff(SA, axis=0)

        return g * (alpha_mid * dCT / dz[:, np.newaxis] -
                    beta_mid  * dSA / dz[:, np.newaxis])

    def depth_from_pressure(
        self, p: Numeric, lat: Optional[Numeric] = None
    ) -> Numeric:
        """
        Depth from sea pressure using the UNESCO (1983) formula with optional
        latitude-dependent gravity (Saunders & Fofonoff, 1976).

        Note: depth is returned as a positive quantity (distance below surface).

        Parameters
        ----------
        p : Numeric
            Sea pressure [dbar]
        lat : Numeric, optional
            Latitude [degrees north]. If not provided, g = 9.81 m/s² is used.

        Returns
        -------
        Numeric
            Depth (positive downward) [m]
        """
        if lat is not None:
            sin2  = np.sin(np.deg2rad(lat)) ** 2
            g_lat = 9.780318 * (1.0 + (5.2792e-3 + 2.36e-5 * sin2) * sin2)
        else:
            g_lat = g

        numer = (9.72659e2 * p - 2.512e-1 * p ** 2 +
                 2.279e-4 * p ** 3 - 1.82e-7 * p ** 4)
        denom = g_lat + 1.092e-6 * p

        return numer / denom

    def pressure_from_depth(
        self, z: Numeric, lat: Optional[Numeric] = None
    ) -> Numeric:
        """
        Sea pressure from depth (positive downward) using a one-step Newton
        refinement of a hydrostatic initial guess.

        Parameters
        ----------
        z : Numeric
            Depth (positive downward) [m]
        lat : Numeric, optional
            Latitude [degrees north]. If not provided, g = 9.81 m/s² is used.

        Returns
        -------
        Numeric
            Sea pressure [dbar]
        """
        if lat is not None:
            sin2  = np.sin(np.deg2rad(lat)) ** 2
            g_lat = 9.780318 * (1.0 + (5.2792e-3 + 2.36e-5 * sin2) * sin2)
        else:
            g_lat = g

        p = z * g_lat / 9.7803 * 1.025  # hydrostatic initial guess

        # One Newton step: p_{n+1} = p_n - (z_from_p(p_n) - z) / dz_dp
        z_est = self.depth_from_pressure(p, lat)
        dzdp  = ((9.72659e2 - 2 * 2.512e-1 * p + 3 * 2.279e-4 * p ** 2
                  - 4 * 1.82e-7 * p ** 3) / (g_lat + 1.092e-6 * p))
        return p + (z - z_est) / dzdp

    # -------------------------------------------------------------------------
    # Derive
    # -------------------------------------------------------------------------

    def derive(self, burst_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute all thermodynamic quantities derivable from the variables present
        in a burst dictionary, and return the burst dictionary augmented with
        those results.

        Each quantity is computed only when all of its required inputs are
        available as keys in ``burst_data``. The method never raises for missing
        inputs — it simply skips any quantities it cannot compute.

        Input keys recognised
        ----------------------
        SP  : Practical Salinity (PSS-78)             [unitless]
        t   : in-situ temperature                       [deg C]
        p   : sea pressure                               [dbar]
        lat : latitude (scalar)     [deg N] -- optional, used for depth

        Output keys added to burst_data
        --------------------------------
        SA          : Absolute Salinity [g/kg]             -- requires SP
        CT          : Conservative Temperature [deg C]     -- requires SA, t, p
        rho         : in-situ density [kg/m³]              -- requires SA, CT, p
        sigma0      : potential density anomaly [kg/m³]    -- requires SA, CT
        alpha       : thermal expansion [1/K]              -- requires SA, CT, p
        beta        : haline contraction [kg/g]            -- requires SA, CT, p
        sound_speed : speed of sound [m/s]                 -- requires SA, CT, p
        t_freezing  : freezing temperature [deg C]         -- requires SA, p
        cp          : isobaric heat capacity [J/(kg K)]    -- requires SA, t, p
        nu          : kinematic viscosity [m²/s]           -- requires t, SA
        N2          : buoyancy frequency² [1/s²]           -- requires SA, CT, p
                      (only computed when n_heights > 1)
        z           : depth (positive downward) [m]        -- requires p

        Parameters
        ----------
        burst_data : dict
            Burst dictionary. Arrays are expected to have shape
            (n_heights, n_samples). Modified in-place and also returned.

        Returns
        -------
        dict
            The input ``burst_data`` dictionary with derived quantities added.
        """
        SP  = burst_data.get("SP")
        t   = burst_data.get("t")
        p   = burst_data.get("p")
        lat = burst_data.get("lat")

        has_SP = SP is not None
        has_t  = t  is not None
        has_p  = p  is not None

        SA = None
        if has_SP:
            SA = self.SA_from_SP(SP)
            burst_data["SA"] = SA

        CT = None
        if SA is not None and has_t and has_p:
            CT = self.CT_from_t(SA, t, p)
            burst_data["CT"] = CT

        if SA is not None and CT is not None and has_p:
            burst_data["rho"]         = self.density(SA, CT, p)
            burst_data["alpha"]       = self.alpha(SA, CT, p)
            burst_data["beta"]        = self.beta(SA, CT, p)
            burst_data["sound_speed"] = self.sound_speed(SA, CT, p)

        if SA is not None and CT is not None:
            burst_data["sigma0"] = self.sigma0(SA, CT)

        if SA is not None and has_p:
            burst_data["t_freezing"] = self.freezing_temperature(SA, p)

        if SA is not None and has_t and has_p:
            burst_data["cp"] = self.heat_capacity(SA, t, p)

        if SA is not None and has_t:
            burst_data["nu"] = self.kinematic_viscosity(t, SA)

        if SA is not None and CT is not None and has_p and self.n_heights > 1:
            burst_data["N2"] = self.buoyancy_frequency(SA, CT, p)

        if has_p:
            burst_data["z"] = self.depth_from_pressure(p, lat)

        return burst_data

    @property
    def data_keys(self):
        return [k for k in self.name_map if k != "time"]
