import numpy as np
from typing import Optional, TypeAlias

from src.utils.constants import (
    GRAVITATIONAL_ACCELERATION as g,
    SSO,
    T0,
    CP0,
)

Numeric: TypeAlias = float | int | np.ndarray


# =============================================================================
# Private helper functions for Conservative Temperature (ct)
# Direct Python translation of Matlab GSW Toolbox (TEOS-10).
# Sources: gsw_entropy_part.m, gsw_entropy_part_zerop.m,
#          gsw_gibbs_pt0_pt0.m, gsw_pt0_from_t.m, gsw_ct_from_pt.m
# =============================================================================


def _entropy_part(sa: Numeric, t: Numeric, p: Numeric) -> Numeric:
    """
    Entropy minus the terms that are a function of only sa (gsw_entropy_part.m).
    """
    sa = np.maximum(sa, 0)
    sfac = 0.0248826675584615
    x2 = sfac * sa
    x = np.sqrt(x2)
    y = t * 0.025
    z = p * 1e-4

    g03 = z * (
        -270.983805184062
        + z * (776.153611613101 + z * (-196.51255088122 + (28.9796526294175 - 2.13290083518327 * z) * z))
    ) + y * (
        -24715.571866078
        + z
        * (
            2910.0729080936
            + z * (-1513.116771538718 + z * (546.959324647056 + z * (-111.1208127634436 + 8.68841343834394 * z)))
        )
        + y
        * (
            2210.2236124548363
            + z
            * (
                -2017.52334943521
                + z * (1498.081172457456 + z * (-718.6359919632359 + (146.4037555781616 - 4.9892131862671505 * z) * z))
            )
            + y
            * (
                -592.743745734632
                + z * (1591.873781627888 + z * (-1207.261522487504 + (608.785486935364 - 105.4993508931208 * z) * z))
                + y
                * (
                    290.12956292128547
                    + z * (-973.091553087975 + z * (602.603274510125 + z * (-276.361526170076 + 32.40953340386105 * z)))
                    + y
                    * (
                        -113.90630790850321
                        + y * (21.35571525415769 - 67.41756835751434 * z)
                        + z * (381.06836198507096 + z * (-133.7383902842754 + 49.023632509086724 * z))
                    )
                )
            )
        )
    )

    g08 = x2 * (
        z
        * (
            729.116529735046
            + z * (-343.956902961561 + z * (124.687671116248 + z * (-31.656964386073 + 7.04658803315449 * z)))
        )
        + x
        * (
            x
            * (
                y * (-137.1145018408982 + y * (148.10030845687618 + y * (-68.5590309679152 + 12.4848504784754 * y)))
                - 22.6683558512829 * z
            )
            + z * (-175.292041186547 + (83.1923927801819 - 29.483064349429 * z) * z)
            + y
            * (
                -86.1329351956084
                + z * (766.116132004952 + z * (-108.3834525034224 + 51.2796974779828 * z))
                + y * (-30.0682112585625 - 1380.9597954037708 * z + y * (3.50240264723578 + 938.26075044542 * z))
            )
        )
        + y
        * (
            1760.062705994408
            + y
            * (
                -675.802947790203
                + y
                * (
                    365.7041791005036
                    + y * (-108.30162043765552 + 12.78101825083098 * y)
                    + z * (-1190.914967948748 + (298.904564555024 - 145.9491676006352 * z) * z)
                )
                + z * (2082.7344423998043 + z * (-614.668925894709 + (340.685093521782 - 33.3848202979239 * z) * z))
            )
            + z
            * (
                -1721.528607567954
                + z * (674.819060538734 + z * (-356.629112415276 + (88.4080716616 - 15.84003094423364 * z) * z))
            )
        )
    )

    return -(g03 + g08) * 0.025


def _entropy_part_zerop(sa: Numeric, pt0: Numeric) -> Numeric:
    """Entropy minus the terms that are a function of only sa, evaluated at p=0 dbar (gsw_entropy_part_zerop.m)."""
    sa = np.maximum(sa, 0)
    sfac = 0.0248826675584615
    x2 = sfac * sa
    x = np.sqrt(x2)
    y = pt0 * 0.025

    g03 = y * (
        -24715.571866078
        + y
        * (
            2210.2236124548363
            + y * (-592.743745734632 + y * (290.12956292128547 + y * (-113.90630790850321 + y * 21.35571525415769)))
        )
    )

    g08 = x2 * (
        x
        * (
            x * (y * (-137.1145018408982 + y * (148.10030845687618 + y * (-68.5590309679152 + 12.4848504784754 * y))))
            + y * (-86.1329351956084 + y * (-30.0682112585625 + y * 3.50240264723578))
        )
        + y
        * (
            1760.062705994408
            + y * (-675.802947790203 + y * (365.7041791005036 + y * (-108.30162043765552 + 12.78101825083098 * y)))
        )
    )

    return -(g03 + g08) * 0.025


def _gibbs_pt0_pt0(sa: Numeric, pt0: Numeric) -> Numeric:
    """Second derivative of specific Gibbs function w.r.t. temperature at p=0 (gsw_gibbs_pt0_pt0.m)."""
    sa = np.maximum(sa, 0)
    sfac = 0.0248826675584615
    x2 = sfac * sa
    x = np.sqrt(x2)
    y = pt0 * 0.025

    g03 = -24715.571866078 + y * (
        4420.4472249096725
        + y * (-1778.231237203896 + y * (1160.5182516851419 + y * (-569.531539542516 + y * 128.13429152494615)))
    )

    g08 = x2 * (
        1760.062705994408
        + x
        * (
            -86.1329351956084
            + x * (-137.1145018408982 + y * (296.20061691375236 + y * (-205.67709290374563 + 49.9394019139016 * y)))
            + y * (-60.136422517125 + y * 10.50720794170734)
        )
        + y * (-1351.605895580406 + y * (1097.1125373015109 + y * (-433.20648175062206 + 63.905091254154904 * y)))
    )

    return (g03 + g08) * 0.000625


def _pt0_from_t(sa: Numeric, t: Numeric, p: Numeric) -> Numeric:
    """
    Potential temperature with p_ref=0 dbar via 2-iteration Newton's method (gsw_pt0_from_t.m).

    Maximum error: 1.8e-14 dec C over the full oceanographic funnel.
    """
    sa = np.maximum(sa, 0)
    s1 = sa * (35.0 / SSO)

    pt0 = t + p * (
        8.65483913395442e-6
        - s1 * 1.41636299744881e-6
        - p * 7.38286467135737e-9
        + t * (-8.38241357039698e-6 + s1 * 2.83933368585534e-8 + t * 1.77803965218656e-8 + p * 1.71155619208233e-10)
    )

    dentropy_dt = CP0 / ((T0 + pt0) * (1.0 - 0.05 * (1.0 - sa / SSO)))
    true_entropy_part = _entropy_part(sa, t, p)

    for _ in range(2):
        pt0_old = pt0
        dentropy = _entropy_part_zerop(sa, pt0_old) - true_entropy_part
        pt0 = pt0_old - dentropy / dentropy_dt
        pt0m = 0.5 * (pt0 + pt0_old)
        dentropy_dt = -_gibbs_pt0_pt0(sa, pt0m)
        pt0 = pt0_old - dentropy / dentropy_dt

    return pt0


def _ct_from_pt(sa: Numeric, pt: Numeric) -> Numeric:
    """
    Conservative Temperature from potential temperature via potential enthalpy
    polynomial (gsw_ct_from_pt.m).
    """
    sa = np.maximum(sa, 0)
    sfac = 0.0248826675584615
    x2 = sfac * sa
    x = np.sqrt(x2)
    y = pt * 0.025

    pot_enthalpy = (
        61.01362420681071
        + y
        * (
            168776.46138048015
            + y
            * (
                -2735.2785605119625
                + y
                * (
                    2574.2164453821433
                    + y
                    * (-1536.6644434977543 + y * (545.7340497931629 + (-50.91091728474331 - 18.30489878927802 * y) * y))
                )
            )
        )
        + x2
        * (
            268.5520265845071
            + y
            * (
                -12019.028203559312
                + y
                * (
                    3734.858026725145
                    + y
                    * (
                        -2046.7671145057618
                        + y * (465.28655623826234 + (-0.6370820302376359 - 10.650848542359153 * y) * y)
                    )
                )
            )
            + x
            * (
                937.2099110620707
                + y * (588.1802812170108 + y * (248.39476522971285 + (-3.871557904936333 - 2.6268019854268356 * y) * y))
                + x
                * (
                    -1687.914374187449
                    + x * (246.9598888781377 + x * (123.59576582457964 - 48.5891069025409 * x))
                    + y
                    * (
                        936.3206544460336
                        + y
                        * (
                            -942.7827304544439
                            + y * (369.4389437509002 + (-33.83664947895248 - 9.987880382780322 * y) * y)
                        )
                    )
                )
            )
        )
    )

    return 2.505092880681252e-4 * pot_enthalpy  # = pot_enthalpy / CP0


def _eos_vars(sa: Numeric, ct: Numeric, p: Numeric):
    """
    Compute normalised coordinates for the 75-term EOS (Roquet et al., 2015).

    Returns (xs, ys, z) where:
        xs = sqrt(sfac * sa + offset),  sfac = 1/(40*(35.16504/35))
        ys = ct * 0.025
        z  = p  * 1e-4
    """
    sfac = 0.0248826675584615  # 1 / (40 * (35.16504/35))
    offset = 5.971840214030754e-1  # deltaS * sfac,  deltaS = 24
    xs = np.sqrt(sfac * np.maximum(sa, 0) + offset)
    ys = ct * 0.025
    z = p * 1e-4
    return xs, ys, z


# =============================================================================
# Public thermodynamic functions
# =============================================================================


def sa_from_sp(sp: Numeric) -> Numeric:
    """
    Absolute Salinity from Practical Salinity using the constant-ratio
    approximation (gsw_sa_from_sp.m, simplified).

    Uses sa = sp × (35.16504 / 35), which skips the geographic Absolute
    Salinity Anomaly (SAAR) correction. Typical error is  ~0.01 g/kg in
    the open ocean. Errors can reach ~0.1 g/kg in marginal seas (Baltic,
    Red Sea, Arctic shelf) where SAAR is significant.

    Parameters
    ----------
    sp : Numeric
        Practical Salinity (PSS-78) [unitless]

    Returns
    -------
    Numeric
        Absolute Salinity [g/kg]
    """
    return np.maximum(sp, 0) * (SSO / 35.0)


def ct_from_t(sa: Numeric, t: Numeric, p: Numeric) -> Numeric:
    """
    Conservative Temperature from in-situ temperature (gsw_ct_from_t.m).

    Computes potential temperature at p_ref = 0 dbar via two iterations of
    Newton's method using Gibbs-entropy polynomials, then converts to
    Conservative Temperature via the potential-enthalpy polynomial.

    Parameters
    ----------
    sa : Numeric
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
    pt0 = _pt0_from_t(sa, t, p)
    return _ct_from_pt(sa, pt0)


# -------------------------------------------------------------------------
# 75-term equation of state (Roquet et al., 2015)
# Coefficients and polynomial structure match gsw_specvol.m / gsw_rho.m /
# gsw_alpha.m / gsw_beta.m / gsw_sound_speed.m / gsw_sigma0.m exactly.
# -------------------------------------------------------------------------


def specific_volume(sa: Numeric, ct: Numeric, p: Numeric) -> Numeric:
    """
    Specific volume from the 75-term polynomial EOS (gsw_specvol.m).

    Parameters
    ----------
    sa : Numeric
        Absolute Salinity [g/kg]
    ct : Numeric
        Conservative Temperature [deg C]
    p : Numeric
        Sea pressure [dbar]

    Returns
    -------
    Numeric
        Specific volume [m³/kg]
    """
    xs, ys, z = _eos_vars(sa, ct, p)

    v000 = 1.0769995862e-3
    v001 = -6.0799143809e-5
    v002 = 9.9856169219e-6
    v003 = -1.1309361437e-6
    v004 = 1.0531153080e-7
    v005 = -1.2647261286e-8
    v006 = 1.9613503930e-9
    v010 = -1.5649734675e-5
    v011 = 1.8505765429e-5
    v012 = -1.1736386731e-6
    v013 = -3.6527006553e-7
    v014 = 3.1454099902e-7
    v020 = 2.7762106484e-5
    v021 = -1.1716606853e-5
    v022 = 2.1305028740e-6
    v023 = 2.8695905159e-7
    v030 = -1.6521159259e-5
    v031 = 7.9279656173e-6
    v032 = -4.6132540037e-7
    v040 = 6.9111322702e-6
    v041 = -3.4102187482e-6
    v042 = -6.3352916514e-8
    v050 = -8.0539615540e-7
    v051 = 5.0736766814e-7
    v060 = 2.0543094268e-7
    v100 = -3.1038981976e-4
    v101 = 2.4262468747e-5
    v102 = -5.8484432984e-7
    v103 = 3.6310188515e-7
    v104 = -1.1147125423e-7
    v110 = 3.5009599764e-5
    v111 = -9.5677088156e-6
    v112 = -5.5699154557e-6
    v113 = -2.7295696237e-7
    v120 = -3.7435842344e-5
    v121 = -2.3678308361e-7
    v122 = 3.9137387080e-7
    v130 = 2.4141479483e-5
    v131 = -3.4558773655e-6
    v132 = 7.7618888092e-9
    v140 = -8.7595873154e-6
    v141 = 1.2956717783e-6
    v150 = -3.3052758900e-7
    v200 = 6.6928067038e-4
    v201 = -3.4792460974e-5
    v202 = -4.8122251597e-6
    v203 = 1.6746303780e-8
    v210 = -4.3592678561e-5
    v211 = 1.1100834765e-5
    v212 = 5.4620748834e-6
    v220 = 3.5907822760e-5
    v221 = 2.9283346295e-6
    v222 = -6.5731104067e-7
    v230 = -1.4353633048e-5
    v231 = 3.1655306078e-7
    v240 = 4.3703680598e-6
    v300 = -8.5047933937e-4
    v301 = 3.7470777305e-5
    v302 = 4.9263106998e-6
    v310 = 3.4532461828e-5
    v311 = -9.8447117844e-6
    v312 = -1.3544185627e-6
    v320 = -1.8698584187e-5
    v321 = -4.8826139200e-7
    v330 = 2.2863324556e-6
    v400 = 5.8086069943e-4
    v401 = -1.7322218612e-5
    v402 = -1.7811974727e-6
    v410 = -1.1959409788e-5
    v411 = 2.5909225260e-6
    v420 = 3.8595339244e-6
    v500 = -2.1092370507e-4
    v501 = 3.0927427253e-6
    v510 = 1.3864594581e-6
    v600 = 3.1932457305e-5

    v = (
        v000
        + xs * (v100 + xs * (v200 + xs * (v300 + xs * (v400 + xs * (v500 + v600 * xs)))))
        + ys
        * (
            v010
            + xs * (v110 + xs * (v210 + xs * (v310 + xs * (v410 + v510 * xs))))
            + ys
            * (
                v020
                + xs * (v120 + xs * (v220 + xs * (v320 + v420 * xs)))
                + ys
                * (
                    v030
                    + xs * (v130 + xs * (v230 + v330 * xs))
                    + ys * (v040 + xs * (v140 + v240 * xs) + ys * (v050 + v150 * xs + v060 * ys))
                )
            )
        )
        + z
        * (
            v001
            + xs * (v101 + xs * (v201 + xs * (v301 + xs * (v401 + v501 * xs))))
            + ys
            * (
                v011
                + xs * (v111 + xs * (v211 + xs * (v311 + v411 * xs)))
                + ys
                * (
                    v021
                    + xs * (v121 + xs * (v221 + v321 * xs))
                    + ys * (v031 + xs * (v131 + v231 * xs) + ys * (v041 + v141 * xs + v051 * ys))
                )
            )
            + z
            * (
                v002
                + xs * (v102 + xs * (v202 + xs * (v302 + v402 * xs)))
                + ys
                * (
                    v012
                    + xs * (v112 + xs * (v212 + v312 * xs))
                    + ys * (v022 + xs * (v122 + v222 * xs) + ys * (v032 + v132 * xs + v042 * ys))
                )
                + z
                * (
                    v003
                    + xs * (v103 + v203 * xs)
                    + ys * (v013 + v113 * xs + v023 * ys)
                    + z * (v004 + v104 * xs + v014 * ys + z * (v005 + v006 * z))
                )
            )
        )
    )

    return v


def density(sa: Numeric, ct: Numeric, p: Numeric) -> Numeric:
    """
    In-situ density from the 75-term polynomial EOS (gsw_rho.m).

    Parameters
    ----------
    sa : Numeric
        Absolute Salinity [g/kg]
    ct : Numeric
        Conservative Temperature [deg C]
    p : Numeric
        Sea pressure [dbar]

    Returns
    -------
    Numeric
        In-situ density [kg/m³]
    """
    return 1.0 / specific_volume(sa, ct, p)


def alpha(sa: Numeric, ct: Numeric, p: Numeric) -> Numeric:
    """
    Thermal expansion coefficient with respect to Conservative Temperature
    from the 75-term polynomial EOS (gsw_alpha.m).

    Parameters
    ----------
    sa : Numeric
        Absolute Salinity [g/kg]
    ct : Numeric
        Conservative Temperature [deg C]
    p : Numeric
        Sea pressure [dbar]

    Returns
    -------
    Numeric
        Thermal expansion coefficient [1/K]
    """
    xs, ys, z = _eos_vars(sa, ct, p)

    a000 = -1.5649734675e-5
    a001 = 1.8505765429e-5
    a002 = -1.1736386731e-6
    a003 = -3.6527006553e-7
    a004 = 3.1454099902e-7
    a010 = 5.5524212968e-5
    a011 = -2.3433213706e-5
    a012 = 4.2610057480e-6
    a013 = 5.7391810318e-7
    a020 = -4.9563477777e-5
    a021 = 2.37838968519e-5
    a022 = -1.38397620111e-6
    a030 = 2.76445290808e-5
    a031 = -1.36408749928e-5
    a032 = -2.53411666056e-7
    a040 = -4.0269807770e-6
    a041 = 2.5368383407e-6
    a050 = 1.23258565608e-6
    a100 = 3.5009599764e-5
    a101 = -9.5677088156e-6
    a102 = -5.5699154557e-6
    a103 = -2.7295696237e-7
    a110 = -7.4871684688e-5
    a111 = -4.7356616722e-7
    a112 = 7.8274774160e-7
    a120 = 7.2424438449e-5
    a121 = -1.03676320965e-5
    a122 = 2.32856664276e-8
    a130 = -3.50383492616e-5
    a131 = 5.1826871132e-6
    a140 = -1.6526379450e-6
    a200 = -4.3592678561e-5
    a201 = 1.1100834765e-5
    a202 = 5.4620748834e-6
    a210 = 7.1815645520e-5
    a211 = 5.8566692590e-6
    a212 = -1.31462208134e-6
    a220 = -4.3060899144e-5
    a221 = 9.4965918234e-7
    a230 = 1.74814722392e-5
    a300 = 3.4532461828e-5
    a301 = -9.8447117844e-6
    a302 = -1.3544185627e-6
    a310 = -3.7397168374e-5
    a311 = -9.7652278400e-7
    a320 = 6.8589973668e-6
    a400 = -1.1959409788e-5
    a401 = 2.5909225260e-6
    a410 = 7.7190678488e-6
    a500 = 1.3864594581e-6

    v_ct_part = (
        a000
        + xs * (a100 + xs * (a200 + xs * (a300 + xs * (a400 + a500 * xs))))
        + ys
        * (
            a010
            + xs * (a110 + xs * (a210 + xs * (a310 + a410 * xs)))
            + ys
            * (
                a020
                + xs * (a120 + xs * (a220 + a320 * xs))
                + ys * (a030 + xs * (a130 + a230 * xs) + ys * (a040 + a140 * xs + a050 * ys))
            )
        )
        + z
        * (
            a001
            + xs * (a101 + xs * (a201 + xs * (a301 + a401 * xs)))
            + ys
            * (
                a011
                + xs * (a111 + xs * (a211 + a311 * xs))
                + ys * (a021 + xs * (a121 + a221 * xs) + ys * (a031 + a131 * xs + a041 * ys))
            )
            + z
            * (
                a002
                + xs * (a102 + xs * (a202 + a302 * xs))
                + ys * (a012 + xs * (a112 + a212 * xs) + ys * (a022 + a122 * xs + a032 * ys))
                + z * (a003 + a103 * xs + a013 * ys + a004 * z)
            )
        )
    )

    return 0.025 * v_ct_part / specific_volume(sa, ct, p)


def beta(sa: Numeric, ct: Numeric, p: Numeric) -> Numeric:
    """
    Haline contraction coefficient at constant Conservative Temperature
    from the 75-term polynomial EOS (gsw_beta.m).

    Parameters
    ----------
    sa : Numeric
        Absolute Salinity [g/kg]
    ct : Numeric
        Conservative Temperature [deg C]
    p : Numeric
        Sea pressure [dbar]

    Returns
    -------
    Numeric
        Haline contraction coefficient [kg/g]
    """
    xs, ys, z = _eos_vars(sa, ct, p)
    sfac = 0.0248826675584615

    b000 = -3.1038981976e-4
    b001 = 2.4262468747e-5
    b002 = -5.8484432984e-7
    b003 = 3.6310188515e-7
    b004 = -1.1147125423e-7
    b010 = 3.5009599764e-5
    b011 = -9.5677088156e-6
    b012 = -5.5699154557e-6
    b013 = -2.7295696237e-7
    b020 = -3.7435842344e-5
    b021 = -2.3678308361e-7
    b022 = 3.9137387080e-7
    b030 = 2.4141479483e-5
    b031 = -3.4558773655e-6
    b032 = 7.7618888092e-9
    b040 = -8.7595873154e-6
    b041 = 1.2956717783e-6
    b050 = -3.3052758900e-7
    b100 = 1.33856134076e-3
    b101 = -6.9584921948e-5
    b102 = -9.62445031940e-6
    b103 = 3.3492607560e-8
    b110 = -8.7185357122e-5
    b111 = 2.2201669530e-5
    b112 = 1.09241497668e-5
    b120 = 7.1815645520e-5
    b121 = 5.8566692590e-6
    b122 = -1.31462208134e-6
    b130 = -2.8707266096e-5
    b131 = 6.3310612156e-7
    b140 = 8.7407361196e-6
    b200 = -2.55143801811e-3
    b201 = 1.12412331915e-4
    b202 = 1.47789320994e-5
    b210 = 1.03597385484e-4
    b211 = -2.95341353532e-5
    b212 = -4.0632556881e-6
    b220 = -5.6095752561e-5
    b221 = -1.4647841760e-6
    b230 = 6.8589973668e-6
    b300 = 2.32344279772e-3
    b301 = -6.9288874448e-5
    b302 = -7.1247898908e-6
    b310 = -4.7837639152e-5
    b311 = 1.0363690104e-5
    b320 = 1.54381356976e-5
    b400 = -1.05461852535e-3
    b401 = 1.54637136265e-5
    b410 = 6.9322972905e-6
    b500 = 1.9159474383e-4

    v_sa_part = (
        b000
        + xs * (b100 + xs * (b200 + xs * (b300 + xs * (b400 + b500 * xs))))
        + ys
        * (
            b010
            + xs * (b110 + xs * (b210 + xs * (b310 + b410 * xs)))
            + ys
            * (
                b020
                + xs * (b120 + xs * (b220 + b320 * xs))
                + ys * (b030 + xs * (b130 + b230 * xs) + ys * (b040 + b140 * xs + b050 * ys))
            )
        )
        + z
        * (
            b001
            + xs * (b101 + xs * (b201 + xs * (b301 + b401 * xs)))
            + ys
            * (
                b011
                + xs * (b111 + xs * (b211 + b311 * xs))
                + ys * (b021 + xs * (b121 + b221 * xs) + ys * (b031 + b131 * xs + b041 * ys))
            )
            + z
            * (
                b002
                + xs * (b102 + xs * (b202 + b302 * xs))
                + ys * (b012 + xs * (b112 + b212 * xs) + ys * (b022 + b122 * xs + b032 * ys))
                + z * (b003 + b103 * xs + b013 * ys + b004 * z)
            )
        )
    )

    return -v_sa_part * 0.5 * sfac / (specific_volume(sa, ct, p) * xs)


def sound_speed(sa: Numeric, ct: Numeric, p: Numeric) -> Numeric:
    """
    Speed of sound in seawater from the 75-term polynomial EOS (gsw_sound_speed.m).

    Parameters
    ----------
    sa : Numeric
        Absolute Salinity [g/kg]
    ct : Numeric
        Conservative Temperature [deg C]
    p : Numeric
        Sea pressure [dbar]

    Returns
    -------
    Numeric
        Speed of sound [m/s]
    """
    xs, ys, z = _eos_vars(sa, ct, p)

    c000 = -6.0799143809e-5
    c001 = 1.99712338438e-5
    c002 = -3.3928084311e-6
    c003 = 4.2124612320e-7
    c004 = -6.3236306430e-8
    c005 = 1.1768102358e-8
    c010 = 1.8505765429e-5
    c011 = -2.3472773462e-6
    c012 = -1.09581019659e-6
    c013 = 1.25816399608e-6
    c020 = -1.1716606853e-5
    c021 = 4.2610057480e-6
    c022 = 8.6087715477e-7
    c030 = 7.9279656173e-6
    c031 = -9.2265080074e-7
    c040 = -3.4102187482e-6
    c041 = -1.26705833028e-7
    c050 = 5.0736766814e-7
    c100 = 2.4262468747e-5
    c101 = -1.16968865968e-6
    c102 = 1.08930565545e-6
    c103 = -4.4588501692e-7
    c110 = -9.5677088156e-6
    c111 = -1.11398309114e-5
    c112 = -8.1887088711e-7
    c120 = -2.3678308361e-7
    c121 = 7.8274774160e-7
    c130 = -3.4558773655e-6
    c131 = 1.55237776184e-8
    c140 = 1.2956717783e-6
    c200 = -3.4792460974e-5
    c201 = -9.6244503194e-6
    c202 = 5.0238911340e-8
    c210 = 1.1100834765e-5
    c211 = 1.09241497668e-5
    c220 = 2.9283346295e-6
    c221 = -1.31462208134e-6
    c230 = 3.1655306078e-7
    c300 = 3.7470777305e-5
    c301 = 9.8526213996e-6
    c310 = -9.8447117844e-6
    c311 = -2.7088371254e-6
    c320 = -4.8826139200e-7
    c400 = -1.7322218612e-5
    c401 = -3.5623949454e-6
    c410 = 2.5909225260e-6
    c500 = 3.0927427253e-6

    v_p = (
        c000
        + xs * (c100 + xs * (c200 + xs * (c300 + xs * (c400 + c500 * xs))))
        + ys
        * (
            c010
            + xs * (c110 + xs * (c210 + xs * (c310 + c410 * xs)))
            + ys
            * (
                c020
                + xs * (c120 + xs * (c220 + c320 * xs))
                + ys * (c030 + xs * (c130 + c230 * xs) + ys * (c040 + c140 * xs + c050 * ys))
            )
        )
        + z
        * (
            c001
            + xs * (c101 + xs * (c201 + xs * (c301 + c401 * xs)))
            + ys
            * (
                c011
                + xs * (c111 + xs * (c211 + c311 * xs))
                + ys * (c021 + xs * (c121 + c221 * xs) + ys * (c031 + c131 * xs + c041 * ys))
            )
            + z
            * (
                c002
                + xs * (c102 + c202 * xs)
                + ys * (c012 + c112 * xs + c022 * ys)
                + z * (c003 + c103 * xs + c013 * ys + z * (c004 + c005 * z))
            )
        )
    )

    v = specific_volume(sa, ct, p)
    return 10000.0 * np.sqrt(-(v**2) / v_p)


def sigma0(sa: Numeric, ct: Numeric) -> Numeric:
    """
    Potential density anomaly referenced to 0 dbar from the 75-term EOS
    (gsw_sigma0.m). Equal to potential density minus 1000 kg/m³.

    Parameters
    ----------
    sa : Numeric
        Absolute Salinity [g/kg]
    ct : Numeric
        Conservative Temperature [deg C]

    Returns
    -------
    Numeric
        Potential density anomaly [kg/m³]
    """
    sfac = 0.0248826675584615
    offset = 5.971840214030754e-1
    xs = np.sqrt(sfac * np.maximum(sa, 0) + offset)
    ys = ct * 0.025

    # 75-term polynomial evaluated at p=0 (z=0); all z-dependent terms drop out.
    # Only the non-commented coefficients from gsw_sigma0.m are needed.
    v000 = 1.0769995862e-3
    v010 = -1.5649734675e-5
    v020 = 2.7762106484e-5
    v030 = -1.6521159259e-5
    v040 = 6.9111322702e-6
    v050 = -8.0539615540e-7
    v060 = 2.0543094268e-7
    v100 = -3.1038981976e-4
    v110 = 3.5009599764e-5
    v120 = -3.7435842344e-5
    v130 = 2.4141479483e-5
    v140 = -8.7595873154e-6
    v150 = -3.3052758900e-7
    v200 = 6.6928067038e-4
    v210 = -4.3592678561e-5
    v220 = 3.5907822760e-5
    v230 = -1.4353633048e-5
    v240 = 4.3703680598e-6
    v300 = -8.5047933937e-4
    v310 = 3.4532461828e-5
    v320 = -1.8698584187e-5
    v330 = 2.2863324556e-6
    v400 = 5.8086069943e-4
    v410 = -1.1959409788e-5
    v420 = 3.8595339244e-6
    v500 = -2.1092370507e-4
    v510 = 1.3864594581e-6
    v600 = 3.1932457305e-5

    vp0 = (
        v000
        + xs * (v100 + xs * (v200 + xs * (v300 + xs * (v400 + xs * (v500 + v600 * xs)))))
        + ys
        * (
            v010
            + xs * (v110 + xs * (v210 + xs * (v310 + xs * (v410 + v510 * xs))))
            + ys
            * (
                v020
                + xs * (v120 + xs * (v220 + xs * (v320 + v420 * xs)))
                + ys
                * (
                    v030
                    + xs * (v130 + xs * (v230 + v330 * xs))
                    + ys * (v040 + xs * (v140 + v240 * xs) + ys * (v050 + v150 * xs + v060 * ys))
                )
            )
        )
    )

    return 1.0 / vp0 - 1000.0


def freezing_temperature(sa: Numeric, p: Numeric) -> Numeric:
    """
    In-situ freezing temperature from a direct polynomial fit (gsw_t_freezing_poly.m).

    Uses the 23-coefficient polynomial given in the comments of gsw_t_freezing_poly.m,
    which avoids calling CT_freezing and t_from_CT. Error is between -8e-4 K and
    +3e-4 K compared with the exact Newton-Raphson method.

    Parameters
    ----------
    sa : Numeric
        Absolute Salinity [g/kg]
    p : Numeric
        Sea pressure [dbar]

    Returns
    -------
    Numeric
        Freezing temperature [deg C]
    """
    sa_r = np.maximum(sa, 0) * 1e-2
    x = np.sqrt(sa_r)
    p_r = p * 1e-4

    c0 = 0.002519
    c1 = -5.946302841607319
    c2 = 4.136051661346983
    c3 = -1.115150523403847e1
    c4 = 1.476878746184548e1
    c5 = -1.088873263630961e1
    c6 = 2.961018839640730
    c7 = -7.433320943962606
    c8 = -1.561578562479883
    c9 = 4.073774363480365e-2
    c10 = 1.158414435887717e-2
    c11 = -4.122639292422863e-1
    c12 = -1.123186915628260e-1
    c13 = 5.715012685553502e-1
    c14 = 2.021682115652684e-1
    c15 = 4.140574258089767e-2
    c16 = -6.034228641903586e-1
    c17 = -1.205825928146808e-2
    c18 = -2.812172968619369e-1
    c19 = 1.877244474023750e-2
    c20 = -1.204395563789007e-1
    c21 = 2.349147739749606e-1
    c22 = 2.748444541144219e-3

    return (
        c0
        + sa_r * (c1 + x * (c2 + x * (c3 + x * (c4 + x * (c5 + c6 * x)))))
        + p_r * (c7 + p_r * (c8 + c9 * p_r))
        + sa_r
        * p_r
        * (
            c10
            + p_r * (c12 + p_r * (c15 + c21 * sa_r))
            + sa_r * (c13 + c17 * p_r + c19 * sa_r)
            + x * (c11 + p_r * (c14 + c18 * p_r) + sa_r * (c16 + c20 * p_r + c22 * sa_r))
        )
    )


def heat_capacity(sa: Numeric, t: Numeric, p: Numeric) -> Numeric:
    """
    Isobaric specific heat capacity of seawater (Fofonoff, 1985, Table 7).

    C_p(S, t, p) = A + B*S + C*S^(3/2)
                 + (D + E*S + F*S^(3/2)) * p
                 + (G + H*S + I*S^(3/2)) * p^2
                 + (J + K*S + M*S^(3/2)) * p^3

    where each letter coefficient is a polynomial in temperature t, and S is
    Practical Salinity (PSS-78), p is in bars.

    Parameters
    ----------
    sa : Numeric
        Absolute Salinity [g/kg]
    t : Numeric
        In-situ temperature [deg C]
    p : Numeric
        Sea pressure [dbar]

    Returns
    -------
    Numeric
        Isobaric heat capacity [J/(kg·K)]

    References
    ----------
    Fofonoff, N.P., 1985: Physical properties of seawater: A new salinity
        scale and equation of state for seawater. J. Geophys. Res., 90,
        3332-3342.
    """
    # Unit conversions: sa [g/kg] → sp [PSS-78],  p [dbar] → p [bar]
    sp = sa * (35.0 / SSO)
    pb = p * 0.1

    A = 4127.4 + t * (-3.720283 + t * (0.1412855 + t * (-2.654387e-3 + t * 2.093236e-5)))

    B = -7.643575 + t * (0.1072763 + t * (-1.38385e-3))

    C = 0.1770383 + t * (-4.07718e-3 + t * (-5.148e-5))

    D = -4.9592e-1 + t * (1.45747e-2 + t * (-3.13885e-4 + t * (2.0357e-6 + t * 1.7168e-8)))

    E = 4.9247e-3 + t * (-1.28315e-4 + t * (9.802e-7 + t * (2.5941e-8 + t * (-2.9179e-10))))

    F = -1.2331e-4 + t * (-1.517e-6 + t * 3.122e-8)

    G = 2.4931e-4 + t * (-1.08645e-5 + t * (2.87533e-7 + t * (-4.0027e-9 + t * 2.2956e-11)))

    H = -2.9558e-6 + t * (1.17054e-7 + t * (-2.3905e-9 + t * 1.8448e-11))

    I = 9.971e-8

    J = -5.422e-8 + t * (2.6380e-9 + t * (-6.5637e-11 + t * 6.136e-13))

    K = 5.540e-10 + t * (-1.7682e-11 + t * 3.513e-13)

    M = -1.4300e-12

    return (
        A
        + B * sp
        + C * sp**1.5
        + (D + E * sp + F * sp**1.5) * pb
        + (G + H * sp + I * sp**1.5) * pb**2
        + (J + K * sp + M * sp**1.5) * pb**3
    )


def dynamic_viscosity(t: Numeric, sa: Numeric) -> Numeric:
    """
    Dynamic viscosity of seawater (Sharqawy et al., 2010).

    Parameters
    ----------
    t : Numeric
        In-situ temperature [deg C]
    sa : Numeric
        Absolute Salinity [g/kg]

    Returns
    -------
    Numeric
        Dynamic viscosity [Pa s]
    """
    mu_w = 4.2844e-5 + 1.0 / (0.157 * (t + 64.993) ** 2 - 91.296)
    S_kg = sa * 1e-3  # g/kg -> kg/kg
    A = 1.541 + 1.998e-2 * t - 9.52e-5 * t**2
    B = 7.974 - 7.561e-2 * t + 4.724e-4 * t**2
    return mu_w * (1.0 + A * S_kg + B * S_kg**2)


def kinematic_viscosity(t: Numeric, sa: Numeric) -> Numeric:
    """
    Kinematic viscosity of seawater.

    Parameters
    ----------
    t : Numeric
        In-situ temperature [deg C]
    sa : Numeric
        Absolute Salinity [g/kg]

    Returns
    -------
    Numeric
        Kinematic viscosity [m²/s]
    """
    ct = ct_from_t(sa, t, np.zeros_like(t))
    rho = density(sa, ct, np.zeros_like(t))
    return dynamic_viscosity(t, sa) / rho


def thermal_conductivity(sa: Numeric, t: Numeric, p: Numeric) -> Numeric:
    """
    Thermal conductivity of seawater (Sharqawy et al., 2010, Eq. 14).

    Parameters
    ----------
    sa : Numeric
        Absolute Salinity [g/kg]
    t : Numeric
        In-situ temperature [deg C]
    p : Numeric
        Sea pressure [dbar]

    Returns
    -------
    Numeric
        Thermal conductivity [W/(m K)]
    """
    p_mpa = p * 0.01  # dbar -> MPa
    k_mw_mk = 0.5715 * (1 + 0.003 * t - 1.025e-5 * t**2 + 6.53e-3 * p_mpa - 2.9e-4 * sa)  # mW/m K
    k_w_mk = k_mw_mk * 1000
    return k_w_mk


def buoyancy_frequency(
    sa: np.ndarray,
    ct: np.ndarray,
    p: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """
    Squared buoyancy (Brunt-Väisälä) frequency from a vertical profile.

    Computed via finite differences of Conservative Temperature and Absolute
    Salinity using the thermal expansion and haline contraction coefficients
    from the 75-term EOS:

        N^2 = g × (alpha × dct/dz − beta × dsa/dz)

    where z is positive upward. N^2 is evaluated at mid-points between
    adjacent instrument depths, so the output has length n_heights − 1
    along axis 0.

    Parameters
    ----------
    sa : np.ndarray
        Absolute Salinity, shape (n_heights, n_samples) [g/kg]
    ct : np.ndarray
        Conservative Temperature, shape (n_heights, n_samples) [deg C]
    p : np.ndarray
        Sea pressure, shape (n_heights, n_samples) [dbar]
    z : np.ndarray
        Instrument depths (positive upward), shape (n_heights,) [m]

    Returns
    -------
    np.ndarray
        N² at mid-depth levels, shape (n_heights − 1, n_samples) [1/s²]
    """
    sa_mid = 0.5 * (sa[:-1] + sa[1:])
    ct_mid = 0.5 * (ct[:-1] + ct[1:])
    p_mid = 0.5 * (p[:-1] + p[1:])

    alpha_mid = alpha(sa_mid, ct_mid, p_mid)
    beta_mid = beta(sa_mid, ct_mid, p_mid)

    dz = np.diff(z)  # shape (n_heights-1,)
    dct = np.diff(ct, axis=0)
    dsa = np.diff(sa, axis=0)

    return g * (alpha_mid * dct / dz[:, np.newaxis] - beta_mid * dsa / dz[:, np.newaxis])


def depth_from_pressure(p: Numeric, lat: Optional[Numeric] = None) -> Numeric:
    """
    Depth from sea pressure using the UNESCO (1983) formula with optional
    latitude-dependent gravity (Saunders & Fofonoff, 1976).

    Note: depth is returned as a positive quantity (distance below surface). If this function is used to
    populate self.z (positive upward), then the depths returned by this function should be multiplied by -1.

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
        sin2 = np.sin(np.deg2rad(lat)) ** 2
        g_lat = 9.780318 * (1.0 + 5.2788e-3 * sin2 + 2.36e-5 * sin2**2)
    else:
        g_lat = g

    depth = (9.72659 * p - 2.2512e-5 * p**2 + 2.279e-10 * p**3 - 1.82e-15 * p**4) / (g_lat + 1.092e-6 * p)

    return depth


def pressure_from_depth(z: Numeric, lat: Optional[Numeric] = None) -> Numeric:
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
        sin2 = np.sin(np.deg2rad(lat)) ** 2
        g_lat = 9.780318 * (1.0 + 5.2788e-3 * sin2 + 2.36e-5 * sin2**2)
    else:
        g_lat = g

    p = z * g_lat / 9.7803 * 1.025  # hydrostatic initial guess

    # One Newton step
    z_est = depth_from_pressure(p, lat)
    numer = 9.72659 * p - 2.2512e-5 * p**2 + 2.279e-10 * p**3 - 1.82e-15 * p**4
    numer_prime = 9.72659 - 2 * 2.2512e-5 * p + 3 * 2.279e-10 * p**2 - 4 * 11.82e-15 * p**3
    denom = g_lat + 1.092e-6 * p
    denom_prime = 1.092e-6
    dzdp = (numer_prime * denom - numer * denom_prime) / denom**2
    return p + (z - z_est) / dzdp
