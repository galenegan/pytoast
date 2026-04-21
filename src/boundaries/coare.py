import numpy as np
from typing import Tuple, Optional, Dict, TypeAlias
from utils.constants import (
    SSO,
    VON_KARMAN as KAPPA,
    STEFAN_BOLTZMANN as SB,
    _PAYNE_TABLE,
    T0,
    GAS_CONSTANT_DRY_AIR as R_AIR,
)
import utils.air_thermo as at
import utils.sea_thermo as st

Numeric: TypeAlias = float | int | np.ndarray

# Gustiness scaling coefficient
BETA = 1.2

# Ratio of molecular weight of water to dry air (used in stability calculations)
EPSILON = 0.61

# Sea ice aerodynamic roughness length [m]
Z0_ICE = 0.0005

# Number of bulk-loop iterations
N_ITER = 10

# Ratio of scalar (heat/moisture) to momentum transfer coefficient (= 1 for COARE)
SCHMIDT = 1.0


# Private stability/profile functions
def _psit_26(zeta: np.ndarray) -> np.ndarray:
    """
    Monin-Obukhov temperature/humidity profile correction function.
    Matches ``psit_26`` in the COARE 3.6 reference code (Fairall et al. 2003).
    """
    dzeta = np.minimum(50.0, 0.35 * zeta)
    # Stable branch (computed for all points; unstable points overwritten below)
    with np.errstate(invalid="ignore"):
        psi = -((1 + 0.6667 * zeta) ** 1.5 + 0.6667 * (zeta - 14.28) * np.exp(-dzeta) + 8.525)
    # Unstable branch (overwrite stable where zeta < 0)
    k = zeta < 0
    x = (1 - 15 * zeta[k]) ** 0.5
    psik = 2 * np.log((1 + x) / 2)
    x = (1 - 34.15 * zeta[k]) ** 0.3333
    psic = (
        1.5 * np.log((1 + x + x**2) / 3)
        - np.sqrt(3) * np.arctan((1 + 2 * x) / np.sqrt(3))
        + 4 * np.arctan(1) / np.sqrt(3)
    )
    f = zeta[k] ** 2 / (1 + zeta[k] ** 2)
    psi[k] = (1 - f) * psik + f * psic
    return psi


def _psiu_26(zeta: np.ndarray) -> np.ndarray:
    """
    Monin-Obukhov momentum profile correction function for COARE 3.6.
    Stable branch coefficient a=0.7 (Edson et al. 2013).
    Matches ``psiu_26`` in the reference code.
    """
    dzeta = np.minimum(50.0, 0.35 * zeta)
    with np.errstate(invalid="ignore"):
        psi = -(0.7 * zeta + 0.75 * (zeta - 5 / 0.35) * np.exp(-dzeta) + 0.75 * 5 / 0.35)
    k = zeta < 0
    x = (1 - 15 * zeta[k]) ** 0.25
    psik = 2 * np.log((1 + x) / 2) + np.log((1 + x**2) / 2) - 2 * np.arctan(x) + 2 * np.arctan(1)
    x = (1 - 10.15 * zeta[k]) ** 0.3333
    psic = (
        1.5 * np.log((1 + x + x**2) / 3)
        - np.sqrt(3) * np.arctan((1 + 2 * x) / np.sqrt(3))
        + 4 * np.arctan(1) / np.sqrt(3)
    )
    f = zeta[k] ** 2 / (1 + zeta[k] ** 2)
    psi[k] = (1 - f) * psik + f * psic
    return psi


def _psiu_40(zeta: np.ndarray) -> np.ndarray:
    """
    Monin-Obukhov momentum profile correction for first-guess iteration.
    Stable branch coefficient a=1.0 (COARE 4.0 style); unstable uses
    different coefficients than _psiu_26.
    Matches ``psiu_40`` in the reference code.
    """
    dzeta = np.minimum(50.0, 0.35 * zeta)
    with np.errstate(invalid="ignore"):
        psi = -(1.0 * zeta + 0.75 * (zeta - 5 / 0.35) * np.exp(-dzeta) + 0.75 * 5 / 0.35)
    k = zeta < 0
    x = (1 - 18 * zeta[k]) ** 0.25
    psik = 2 * np.log((1 + x) / 2) + np.log((1 + x**2) / 2) - 2 * np.arctan(x) + 2 * np.arctan(1)
    x = (1 - 10 * zeta[k]) ** 0.3333
    psic = (
        1.5 * np.log((1 + x + x**2) / 3)
        - np.sqrt(3) * np.arctan((1 + 2 * x) / np.sqrt(3))
        + 4 * np.arctan(1) / np.sqrt(3)
    )
    f = zeta[k] ** 2 / (1 + zeta[k] ** 2)
    psi[k] = (1 - f) * psik + f * psic
    return psi


def sea_surface_albedo(
    sw_down: Numeric,
    julian_day: Numeric,
    lat: Numeric,
    lon: Numeric,
) -> Tuple[Numeric, Numeric, Numeric, Numeric]:
    """
    Sea-surface albedo from the Payne (1972) look-up table.

    Computes solar geometry from position and time, estimates atmospheric
    transmission from the ratio of measured to clear-sky irradiance, then
    interpolates the Payne (1972) table to obtain albedo. Equivalent to
    ``albedo_vector`` in the COARE 3.6 reference code, rewritten as a fully
    vectorised, idiomatic function.

    Parameters
    ----------
    sw_down : Numeric
        Downwelling shortwave irradiance at the surface (W/m^2).
    julian_day : Numeric
        Julian day (astronomical convention: fractional part encodes UTC time,
        where julian_day - floor(julian_day) = 0 corresponds to noon UTC).
    lat : Numeric
        Latitude (degrees north).
    lon : Numeric
        Longitude (degrees east).

    Returns
    -------
    alb : Numeric
        Sea-surface albedo (-).
    transmission : Numeric
        Atmospheric transmission (-).
    solarmax : Numeric
        Clear-sky (top-of-atmosphere) irradiance at the surface (W/m^2).
    solar_altitude : Numeric
        Solar altitude angle (degrees).

    References
    ----------
    Payne, R. E., 1972: Albedo of the sea surface. J. Atmos. Sci., 29, 959–970.
    """
    sw_down = np.atleast_1d(np.asarray(sw_down, dtype=float))

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)  # positive east

    # UTC hour from fractional Julian day (julian_day=0 at noon)
    utc = (julian_day - np.fix(julian_day)) * 24

    # Solar hour angle (lon_rad positive east -> add to hour-angle term)
    hour_angle = np.pi * utc / 12 + lon_rad

    # Solar declination
    declination_deg = 23.45 * np.cos(2 * np.pi * (julian_day - 173) / 365.25)
    declination_rad = np.deg2rad(declination_deg)

    # Solar altitude angle
    sin_altitude = np.sin(lat_rad) * np.sin(declination_rad) - np.cos(lat_rad) * np.cos(declination_rad) * np.cos(
        hour_angle
    )
    solar_altitude = np.rad2deg(np.arcsin(sin_altitude))

    # Clear-sky surface irradiance
    SC = 1380.0
    solarmax = SC * sin_altitude

    # Atmospheric transmission, clipped to [0, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        transmission = np.where(solarmax > 0, np.minimum(sw_down / solarmax, 2.0), 0.0)

    # Nearest-neighbour lookup into the Payne table
    # Transmission grid: 0, 0.05, ..., 1.00  -> 21 rows
    # Solar altitude grid: 0, 2, ..., 90     -> 46 cols
    i_T = np.clip(np.round(transmission / 0.05).astype(int), 0, 20)
    i_alt = np.clip(np.round(solar_altitude / 2.0).astype(int), 0, 45)
    alb = _PAYNE_TABLE[i_T, i_alt]

    # Night / sun-below-horizon: set all outputs to zero
    night = solar_altitude <= 0
    alb = np.where(night, 0.0, alb)
    transmission = np.where(night, 0.0, transmission)
    solarmax = np.where(night, 0.0, solarmax)
    solar_altitude = np.where(night, 0.0, solar_altitude)

    return alb, transmission, solarmax, solar_altitude


# Main COARE 3.6 function


def coare36(
    u: Numeric,
    z_u: Numeric,
    t: Numeric,
    z_t: Numeric,
    rh: Numeric,
    z_rh: Numeric,
    p: Numeric,
    ts: Numeric,
    sw_down: Numeric,
    lw_down: Numeric,
    julian_day: Numeric,
    lat: Numeric,
    lon: Numeric = 0.0,
    pbl_height: Numeric = 600.0,
    rain: Numeric = 0.0,
    surface_salinity: Numeric = SSO,
    phase_speed: Optional[Numeric] = None,
    h_sig: Optional[Numeric] = None,
    u_surf: Numeric = 0.0,
    zref_u: Numeric = 10.0,
    zref_t: Numeric = 10.0,
    zref_rh: Numeric = 10.0,
) -> Dict[str, np.ndarray]:
    """
    COARE 3.6 bulk air-sea flux algorithm (Fairall et al. 2003, Edson et al. 2013).

    Computes turbulent and radiative fluxes at the air-sea interface from
    mean meteorological measurements. Includes the COARE cool-skin
    parameterization. For warm-layer effects use ``coare36_warm_layer``.

    This is a port of the Python implementation by Ludovic Bariteau which can be found here:
    https://github.com/NOAA-PSL/COARE-algorithm/blob/master/Python/COARE3.6/coare36vn_zrf_et.py. There is no functional
    difference between the two implementations: they produce near-identical output, as verified in
    py-tests/src/boundaries/test_coare.py. The main differences are stylistic, with this implementation being written in
    idiomatic python with dictionary output rather than the MATLAB port style of the original. This implementation is
    also integrated with the air_thermo and sea_thermo PyTOAST modules, which produces small differences in the output.


    Parameters
    ----------
    u : Numeric
        Wind speed relative to the ocean surface (m/s) at height z_u.
    z_u : Numeric
        Height of wind measurement (m).
    t : Numeric
        Air temperature (C) at height z_t.
    z_t : Numeric
        Height of air temperature measurement (m).
    rh : Numeric
        Relative humidity (%) at height z_rh.
    z_rh : Numeric
        Height of relative humidity measurement (m).
    p : Numeric
        Sea-level air pressure (mb).
    ts : Numeric
        Sea surface (bulk or subsurface) temperature (C).
    sw_down : Numeric
        Downwelling shortwave radiation (W/m^2).
    lw_down : Numeric
        Downwelling longwave radiation (W/m^2).
    julian_day : Numeric
        Julian day (fractional part = UTC time; 0 = noon UTC).
    lat : Numeric
        Latitude (degrees north).
    lon : Numeric, optional
        Longitude (degrees east). Default 0.
    pbl_height : Numeric, optional
        Planetary boundary layer height (m). Default 600.
    rain : Numeric, optional
        Rain rate (mm/hr). Default 0.
    surface_salinity : Numeric, optional
        Sea surface salinity (PSU). Default SSO.
    phase_speed : Numeric, optional
        Phase speed of dominant waves (m/s). If given, uses wave-age-based
        Charnock parameterization.
    h_sig : Numeric, optional
        Significant wave height (m). Required with phase_speed.
    u_surf : Numeric, optional
        Ocean surface current speed (m/s). Default 0.
    zref_u : Numeric, optional
        Reference height for wind profile output (m). Default 10.
    zref_t : Numeric, optional
        Reference height for temperature profile output (m). Default 10.
    zref_rh : Numeric, optional
        Reference height for humidity profile output (m). Default 10.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of output variables. Sign convention: sensible, latent, and
        rain heat fluxes are *positive when cooling the ocean*; radiative fluxes
        are *positive when heating the ocean*.

    References
    ----------
    Fairall, C. W., et al., 2003: Bulk parameterization of air-sea fluxes:
        Updates and verification for the COARE algorithm. J. Climate, 16, 571-590.
    Edson, J. B., et al., 2013: On the exchange of momentum over the open ocean.
        J. Phys. Oceanogr., 43, 1589-1610.
    """
    # Ensure all primary inputs are 1-D float arrays
    u = np.atleast_1d(np.asarray(u, dtype=float)).ravel()
    N = u.size

    def _to_array(x, length=N):
        return np.broadcast_to(np.atleast_1d(np.asarray(x, dtype=float)), (length,)).copy()

    z_u = _to_array(z_u)
    t = _to_array(t)
    z_t = _to_array(z_t)
    rh = _to_array(rh)
    z_rh = _to_array(z_rh)
    p = _to_array(p)
    ts = _to_array(ts)
    sw_down = _to_array(sw_down)
    lw_down = _to_array(lw_down)
    julian_day = _to_array(julian_day)
    lat = _to_array(lat)
    lon = _to_array(lon)
    pbl_height = _to_array(pbl_height)
    rain = _to_array(rain)
    surface_salinity = _to_array(surface_salinity)
    u_surf = _to_array(u_surf)

    if phase_speed is not None:
        phase_speed = _to_array(phase_speed)
    else:
        phase_speed = np.full(N, np.nan)

    if h_sig is not None:
        h_sig = _to_array(h_sig)
    else:
        h_sig = np.full(N, np.nan)

    # Freezing point and ice mask
    surface_salinity_abs = st.sa_from_sp(surface_salinity)
    t_freeze = st.freezing_temperature(surface_salinity_abs, p=0)
    idx_ice = ts < t_freeze
    # Cool-skin active only when sea surface is above freezing
    j_cool = np.where(idx_ice, 0.0, 1.0)

    # Humidity at T/q measurement height
    # Pressure at temperature/humidity measurement height (hydrostatic)
    p_tq = p - 0.125 * z_t

    # Air specific humidity [kg/kg] and vapor pressure [mb]
    q_air = at.specific_humidity(t, p_tq, rh)

    # Sea-surface saturation specific humidity [kg/kg]
    q_s = at.saturation_specific_humidity(ts, p, surface_salinity, t_freeze)

    # Thermodynamic constants
    g = st.gravity_at_lat(lat)

    # Specific heat capacity of air
    CP_AIR = at.specific_heat(t)

    # Dry adiabatic lapse rate [K/m] -- positive, temperature decreases with height
    lapse_rate = at.dry_adiabatic_lapse_rate(t, g)

    # Latent heat of vaporization [J/kg] -- evaluated at sea surface temperature
    L_e = at.latent_heat_of_vaporization(ts)

    # Air density at T/q measurement height [kg/m3]
    rho_air = at.air_density(t, p_tq, rh)

    # Kinematic viscosity of air [m2/s]
    nu_air = at.kinematic_viscosity(t)

    # Cool-skin parameterization constants
    # Thermal expansion coefficient of seawater at surface [1/C]
    tsw = np.maximum(ts, t_freeze)  # clamp to freeze for stability
    ts_conservative = st.ct_from_t(surface_salinity_abs, tsw, p=0)
    alpha_sw = st.alpha(surface_salinity_abs, ts_conservative, p=0)

    # Haline expansion contribution to buoyancy in skin layer
    haline_buoyancy_coeff = st.beta(surface_salinity_abs, ts_conservative, p=0)

    # Cool-skin / warm-layer water properties. Using the COARE defaults here rather than TEOS-10
    cp_water = 4000.0  # specific heat of seawater [J/(kg K)]
    rho_water = 1022.0  # density of seawater [kg/m3]
    visw = 1e-6  # kinematic viscosity of seawater [m2/s]
    k_water = 0.6  # thermal conductivity of seawater [W/(m K)]
    # Combined cool-skin parameter
    cool_skin_param = 16 * g * cp_water * (rho_water * visw) ** 3 / (k_water**2 * rho_air**2)
    # Linearized surface humidity–temperature coefficient [kg/(kg K)]
    dqs_dT = 0.622 * L_e * q_s / (R_AIR * (ts + T0) ** 2)

    # Radiative fluxes
    albedo, _, _, _ = sea_surface_albedo(sw_down, julian_day, lat, lon)
    sw_net = (1 - albedo) * sw_down
    # lw_net is positive when cooling the ocean (sign flipped at output)
    lw_net = 0.97 * (SB * (ts - 0.3 * j_cool + T0) ** 4 - lw_down)

    # ── First-guess initial conditions ───────────────────────────────────────
    delta_u = u - u_surf  # air-sea relative wind speed [m/s]
    # Temperature difference including lapse-rate correction
    delta_T = ts - t - lapse_rate * z_t
    delta_q = q_s - q_air  # air-sea specific humidity difference [kg/kg]
    t_kelvin = at.t_c2kelvin(t)  # air temperature [K]

    dT_skin = 0.3  # first-guess cool-skin temperature depression [C]
    gust = 0.5  # first-guess gustiness [m/s]
    u_tot = np.sqrt(delta_u**2 + gust**2)  # total (gustiness-inclusive) wind speed

    # Estimate 10-m wind from log profile with smooth roughness
    u_10 = u_tot * np.log(10 / 1e-4) / np.log(z_u / 1e-4)
    ustar = 0.035 * u_10
    z0_10 = 0.011 * ustar**2 / g + 0.11 * nu_air / ustar
    Cd_10 = (KAPPA / np.log(10 / z0_10)) ** 2
    Ch_10 = 0.00115
    Ct_10 = Ch_10 / np.sqrt(Cd_10)
    z0t_10 = 10 / np.exp(KAPPA / Ct_10)

    # Transfer coefficients at measurement heights using 10-m roughness lengths
    Cd = (KAPPA / np.log(z_u / z0_10)) ** 2
    Ct = KAPPA / np.log(z_t / z0t_10)
    # Coefficient used for relating bulk Richardson number to stability parameter
    zeta_Ri_coeff = KAPPA * Ct / Cd

    # Convective limit for bulk Richardson number
    Ri_bu_conv = -z_u / pbl_height / 0.004 / BETA**3

    # Bulk Richardson number
    Ri_bu = -g * z_u / t_kelvin * ((delta_T - dT_skin * j_cool) + EPSILON * t_kelvin * delta_q) / u_tot**2

    # Initial stability parameter
    zeta_u = zeta_Ri_coeff * Ri_bu * (1 + 27 / 9 * Ri_bu / zeta_Ri_coeff)
    # For unstable conditions: apply convective limit
    unstable = Ri_bu < 0
    if np.size(Ri_bu_conv) == 1:
        zeta_u[unstable] = zeta_Ri_coeff[unstable] * Ri_bu[unstable] / (1 + Ri_bu[unstable] / Ri_bu_conv)
    else:
        zeta_u[unstable] = zeta_Ri_coeff[unstable] * Ri_bu[unstable] / (1 + Ri_bu[unstable] / Ri_bu_conv[unstable])

    # Save indices where stability is very strong (zeta > 50) for special treatment
    idx_very_stable = zeta_u > 50

    L_10 = z_u / zeta_u
    gust_factor = u_tot / delta_u  # ratio of total to relative wind

    # First-guess scaling parameters using psiu_40 stability function
    ustar = u_tot * KAPPA / (np.log(z_u / z0_10) - _psiu_40(z_u / L_10))
    tstar = -(delta_T - dT_skin * j_cool) * KAPPA * SCHMIDT / (np.log(z_t / z0t_10) - _psit_26(z_t / L_10))
    qstar = -(delta_q - dqs_dT * dT_skin * j_cool) * KAPPA * SCHMIDT / (np.log(z_rh / z0t_10) - _psit_26(z_rh / L_10))

    dz_skin = 0.001 * np.ones(N)  # first-guess cool-skin thickness [m]

    # Charnock coefficient setup
    # Wind-speed-dependent Charnock (COARE 3.5 parameterization)
    charn_a1, charn_a2, u10_max = 0.0017, -0.005, 19.0
    charnock = np.clip(charn_a1 * u_10 + charn_a2, None, charn_a1 * u10_max + charn_a2)

    # Wave-age-based Charnock: fill estimated h_sig where cp given but h_sig missing
    wave_Ad, wave_Bd = 0.2, 2.2
    h_sig_est = np.maximum(0.25, (0.02 * (phase_speed / u_10) ** 1.1 - 0.0025) * u_10**2)
    missing_h_sig = ~np.isnan(phase_speed) & np.isnan(h_sig)
    h_sig[missing_h_sig] = h_sig_est[missing_h_sig]

    z0_wave = h_sig * wave_Ad * (ustar / phase_speed) ** wave_Bd
    charnock_wave = z0_wave * g / ustar**2

    # Use wave-age Charnock where wave data are available
    have_waves = ~np.isnan(phase_speed)
    charnock[have_waves] = charnock_wave[have_waves]

    # Store first-iteration values for very-stable fallback (applied after loop)
    ustar_stable = tstar_stable = qstar_stable = None
    L_stable = zeta_stable = dT_skin_stable = dq_skin_stable = dz_skin_stable = None

    # Bulk iteration loop
    for i_iter in range(N_ITER):
        # MO stability parameter at wind measurement height
        zeta = KAPPA * g * z_u / t_kelvin * (tstar + EPSILON * t_kelvin * qstar) / ustar**2
        L = z_u / zeta

        # Roughness lengths
        z0 = charnock * ustar**2 / g + 0.11 * nu_air / ustar
        z0[idx_ice] = Z0_ICE
        Re_rough = z0 * ustar / nu_air  # roughness Reynolds number

        # Scalar roughness lengths (COARE 3.0)
        z0q = np.minimum(1.6e-4, 5.8e-5 / Re_rough**0.72)
        # Over sea ice: Andreas (1987) parameterization
        z0t = z0q.copy()
        if idx_ice.any():
            for ij in np.where(idx_ice)[0]:
                rr_i = Re_rough[ij]
                if rr_i <= 0.135:
                    z0t[ij] = Re_rough[ij] * np.exp(1.25)
                    z0q[ij] = Re_rough[ij] * np.exp(1.61)
                elif rr_i <= 2.5:
                    z0t[ij] = rr_i * np.exp(0.149 - 0.55 * np.log(rr_i))
                    z0q[ij] = rr_i * np.exp(0.351 - 0.628 * np.log(rr_i))
                else:
                    z0t[ij] = rr_i * np.exp(0.317 - 0.565 * np.log(rr_i) - 0.183 * np.log(rr_i) ** 2)
                    z0q[ij] = rr_i * np.exp(0.396 - 0.512 * np.log(rr_i) - 0.18 * np.log(rr_i) ** 2)

        # Log-profile transfer coefficients
        c_momentum = KAPPA / (np.log(z_u / z0) - _psiu_26(z_u / L))
        c_humidity = KAPPA * SCHMIDT / (np.log(z_rh / z0q) - _psit_26(z_rh / L))
        c_heat = KAPPA * SCHMIDT / (np.log(z_t / z0t) - _psit_26(z_t / L))

        # Update scaling parameters
        ustar = u_tot * c_momentum
        qstar = -(delta_q - dqs_dT * dT_skin * j_cool) * c_humidity
        tstar = -(delta_T - dT_skin * j_cool) * c_heat

        # Buoyancy flux scaling (Stull 1988 formulation used in COARE 3.6)
        tstar_virtual = tstar * (1 + EPSILON * q_air) + EPSILON * t_kelvin * qstar
        # Original COARE buoyancy flux (retained for hbb/hsbb outputs)
        tstar_virtual_orig = tstar + EPSILON * t_kelvin * qstar
        tstar_sonic_orig = tstar + 0.51 * t_kelvin * qstar

        # Gustiness (convective velocity scale)
        buoyancy_flux = -g / t_kelvin * ustar * tstar_virtual
        gust = np.where(buoyancy_flux > 0, BETA * (buoyancy_flux * pbl_height) ** 0.333, 0.2)
        u_tot = np.sqrt(delta_u**2 + gust**2)
        gust_factor = u_tot / delta_u

        # Intermediate heat fluxes for cool-skin calculation
        hs_iter = -rho_air * CP_AIR * ustar * tstar
        hl_iter = -rho_air * L_e * ustar * qstar
        q_net = lw_net + hs_iter + hl_iter  # net cooling flux into cool skin

        # Solar absorption within skin layer (Fairall et al. 1996 / Wick et al. 2005)
        sw_absorbed_skin = sw_net * (0.065 + 11 * dz_skin - 6.6e-5 / dz_skin * (1 - np.exp(-dz_skin / 8e-4)))
        q_net_skin = q_net - sw_absorbed_skin

        # Buoyancy flux into skin layer (thermal + haline terms)
        buoyancy_flux_skin = alpha_sw * q_net_skin + haline_buoyancy_coeff * hl_iter * cp_water / L_e

        # Cool-skin thickness
        lambda_cool = 6.0 * np.ones(N)
        dz_skin = np.minimum(0.01, lambda_cool * visw / (np.sqrt(rho_air / rho_water) * ustar))
        warm_skin = buoyancy_flux_skin > 0
        lambda_cool[warm_skin] = (
            6
            / (1 + (cool_skin_param[warm_skin] * buoyancy_flux_skin[warm_skin] / ustar[warm_skin] ** 4) ** 0.75)
            ** 0.333
        )
        dz_skin[warm_skin] = (
            lambda_cool[warm_skin] * visw / (np.sqrt(rho_air[warm_skin] / rho_water) * ustar[warm_skin])
        )
        dT_skin = q_net_skin * dz_skin / k_water
        dq_skin = dqs_dT * dT_skin
        lw_net = 0.97 * (SB * (ts - dT_skin * j_cool + T0) ** 4 - lw_down)

        # Update neutral 10-m wind and Charnock coefficient
        u10N = ustar / KAPPA / gust_factor * np.log(10 / z0)
        charnock = np.clip(charn_a1 * u10N + charn_a2, None, charn_a1 * u10_max + charn_a2)
        # Recompute wave-age Charnock with updated ustar
        z0_wave = h_sig * wave_Ad * (ustar / phase_speed) ** wave_Bd
        charnock_wave = z0_wave * g / ustar**2
        charnock[have_waves] = charnock_wave[have_waves]

        # Save first-iteration values for very-stable fallback
        if i_iter == 0:
            ustar_stable = ustar[idx_very_stable].copy()
            tstar_stable = tstar[idx_very_stable].copy()
            qstar_stable = qstar[idx_very_stable].copy()
            L_stable = L[idx_very_stable].copy()
            zeta_stable = zeta[idx_very_stable].copy()
            dT_skin_stable = dT_skin[idx_very_stable].copy()
            dq_skin_stable = dq_skin[idx_very_stable].copy()
            dz_skin_stable = dz_skin[idx_very_stable].copy()

    # Restore first-iteration values where stability is extreme (zeta > 50)
    ustar[idx_very_stable] = ustar_stable
    tstar[idx_very_stable] = tstar_stable
    qstar[idx_very_stable] = qstar_stable
    L[idx_very_stable] = L_stable
    zeta[idx_very_stable] = zeta_stable
    dT_skin[idx_very_stable] = dT_skin_stable
    dq_skin[idx_very_stable] = dq_skin_stable
    dz_skin[idx_very_stable] = dz_skin_stable

    # Compute output fluxes
    tau = rho_air * ustar**2 / gust_factor  # wind stress [N/m2]
    hs = -rho_air * CP_AIR * ustar * tstar  # sensible heat [W/m2]
    hl = -rho_air * L_e * ustar * qstar  # latent heat [W/m2]
    h_buoyancy = -rho_air * CP_AIR * ustar * tstar_virtual_orig
    h_buoyancy_sonic = -rho_air * CP_AIR * ustar * tstar_sonic_orig

    # Webb correction
    w_bar = 1.61 * hl / L_e / (1 + 1.61 * q_air) / rho_air + hs / rho_air / CP_AIR / t_kelvin
    hl_webb = rho_air * w_bar * q_air * L_e

    evap = 1000 * hl / L_e / 1000 * 3600  # evaporation [mm/hr]

    # Transfer coefficients
    Cd = tau / rho_air / u_tot / np.maximum(0.1, delta_u)
    Ch = -ustar * tstar / u_tot / (delta_T - dT_skin * j_cool)
    Ce = -ustar * qstar / (delta_q - dq_skin * j_cool) / u_tot
    # Neutral transfer coefficients at 10 m
    Cdn10 = KAPPA**2 / np.log(10 / z0) ** 2
    Chn10 = KAPPA**2 * SCHMIDT / (np.log(10 / z0) * np.log(10 / z0t))
    Cen10 = KAPPA**2 * SCHMIDT / (np.log(10 / z0) * np.log(10 / z0q))

    # Reference-height and 10-m profiles
    psi_u = _psiu_26(z_u / L)
    psi_10 = _psiu_26(10.0 / L)
    psi_ref_u = _psiu_26(zref_u / L)
    psi_T = _psit_26(z_t / L)
    psi_10T = _psit_26(10.0 / L)
    psi_ref_T = _psit_26(zref_t / L)
    psi_ref_q = _psit_26(zref_rh / L)

    # Wind speeds
    S_10 = u_tot + ustar / KAPPA * (np.log(10 / z_u) - psi_10 + psi_u)
    U10 = S_10 / gust_factor
    U10N = U10 + psi_10 * ustar / KAPPA / gust_factor
    UN = delta_u + psi_u * ustar / KAPPA / gust_factor
    Urf = delta_u + ustar / KAPPA / gust_factor * (np.log(zref_u / z_u) - psi_ref_u + psi_u)
    UrfN = Urf + psi_ref_u * ustar / KAPPA / gust_factor

    # Temperatures (with lapse-rate height correction)
    P10 = p - 0.125 * 10
    P_ref = p - 0.125 * zref_t
    T10 = t + tstar / KAPPA * (np.log(10 / z_t) - psi_10T + psi_T) + lapse_rate * (z_t - 10)
    Trf = t + tstar / KAPPA * (np.log(zref_t / z_t) - psi_ref_T + psi_T) + lapse_rate * (z_t - zref_t)
    T10N = T10 + psi_10T * tstar / KAPPA
    TrfN = Trf + psi_ref_T * tstar / KAPPA

    # Specific humidities [g/kg]
    dq_skin_output = dqs_dT * dT_skin * j_cool
    q_s_output = (q_s - dq_skin_output) * 1000
    q_air_10_gkg = q_air * 1000 + 1000 * qstar / KAPPA * (np.log(10 / z_rh) - psi_10T + psi_T)
    q_air_rf_gkg = q_air * 1000 + 1000 * qstar / KAPPA * (np.log(zref_rh / z_rh) - psi_ref_q + psi_T)
    Q10N = q_air_10_gkg + psi_10T * 1000 * qstar / KAPPA
    QrfN = q_air_rf_gkg + psi_ref_q * 1000 * qstar / KAPPA

    # Relative humidities at reference heights
    t_freeze_10 = t_freeze  # same salinity -> same freeze point
    RH10 = at.relative_humidity_from_specific_humidity(T10, P10, q_air_10_gkg / 1000, t_freeze_10)
    RHrf = at.relative_humidity_from_specific_humidity(Trf, P_ref, q_air_rf_gkg / 1000, t_freeze_10)

    # Air density at 10 m
    rho_air10 = at.air_density(T10, P10, RH10)

    # Rain heat flux
    diffusivity_water_vapor = 2.11e-5 * ((t + T0) / T0) ** 1.94
    diffusivity_heat = (1 + 3.309e-3 * t - 1.44e-6 * t**2) * 0.02411 / (rho_air * CP_AIR)
    dqs_dT_bulk = q_air * L_e / (R_AIR * (t + T0) ** 2)
    alfac = 1 / (1 + 0.622 * dqs_dT_bulk * L_e * diffusivity_water_vapor / (CP_AIR * diffusivity_heat))
    T_skin = ts - dT_skin * j_cool
    dq_skin_out = dqs_dT * dT_skin * j_cool
    hrain = rain * alfac * cp_water * ((T_skin - t) + (q_s - q_air - dq_skin_out) * L_e / CP_AIR) / 3600

    # Wave breaking statistics
    whitecap_fraction = 0.00073 * (U10N - 2) ** 1.43
    whitecap_fraction[U10 < 2.1] = 1e-5
    whitecap_fraction[have_waves] = (
        0.0016 * U10N[have_waves] ** 1.1 / np.sqrt(phase_speed[have_waves] / U10N[have_waves])
    )
    whitecap_fraction[idx_ice] = 0.0

    wave_break_dissipation = 0.095 * rho_air * U10N * ustar**2
    wave_break_dissipation[idx_ice] = 0.0

    # Output
    # Set NaN outputs for missing wind data
    bad_input = np.isnan(u)
    gust[bad_input] = np.nan
    dz_skin[bad_input] = np.nan
    z0t[bad_input] = np.nan
    z0q[bad_input] = np.nan

    # Apply cool-skin mask (only return non-zero where cool skin was active)
    dT_skin_out = dT_skin * j_cool
    dq_skin_out_gkg = dq_skin_out * 1000  # [g/kg]

    return {
        # Turbulent fluxes (positive = cooling ocean)
        "ustar": ustar,
        "tau": tau,
        "hs": hs,
        "hl": hl,
        "hrain": hrain,
        "h_buoyancy": h_buoyancy,
        "h_buoyancy_sonic": h_buoyancy_sonic,
        "hl_webb": hl_webb,
        # Scaling parameters
        "tstar": tstar,
        "qstar": qstar,
        # Roughness lengths
        "z0": z0,
        "z0t": z0t,
        "z0q": z0q,
        # Transfer coefficients
        "Cd": Cd,
        "Ch": Ch,
        "Ce": Ce,
        "Cdn10": Cdn10,
        "Chn10": Chn10,
        "Cen10": Cen10,
        # Stability
        "L": L,
        "zeta": z_u / L,
        # Cool skin
        "dT_skin": dT_skin_out,
        "dq_skin": dq_skin_out_gkg,
        "dz_skin": dz_skin,
        "T_skin": T_skin,
        # Radiative fluxes (positive = heating ocean)
        "sw_net": sw_net,
        "lw_net": -lw_net,
        # Thermodynamic properties
        "Le": L_e,
        "rho_air": rho_air,
        # Wind profiles
        "U10": U10,
        "U10N": U10N,
        "UN": UN,
        "Urf": Urf,
        "UrfN": UrfN,
        # Temperature profiles
        "T10": T10,
        "T10N": T10N,
        "Trf": Trf,
        "TrfN": TrfN,
        # Humidity profiles [g/kg]
        "Q10": q_air_10_gkg,
        "Q10N": Q10N,
        "Qrf": q_air_rf_gkg,
        "QrfN": QrfN,
        "Qs": q_s_output,
        # Other atmospheric profiles
        "RH10": RH10,
        "RHrf": RHrf,
        "P10": P10,
        "rho_air10": rho_air10,
        # Bulk diagnostics
        "evap": evap,
        "gust": gust,
        "whitecap_fraction": whitecap_fraction,
        "wave_break_dissipation": wave_break_dissipation,
    }


# Warm-layer wrapper


def coare36_warm_layer(
    julian_day: np.ndarray,
    u: np.ndarray,
    z_u: Numeric,
    t: np.ndarray,
    z_t: Numeric,
    rh: np.ndarray,
    z_rh: Numeric,
    p: np.ndarray,
    ts: np.ndarray,
    sw_down: np.ndarray,
    lw_down: np.ndarray,
    lat: Numeric,
    lon: Numeric,
    pbl_height: Numeric,
    rain: np.ndarray,
    ts_depth: np.ndarray,
    surface_salinity: np.ndarray = None,
    phase_speed: Optional[np.ndarray] = None,
    h_sig: Optional[np.ndarray] = None,
    zref_u: Numeric = 10.0,
    zref_t: Numeric = 10.0,
    zref_rh: Numeric = 10.0,
) -> Dict[str, np.ndarray]:
    """
    COARE 3.6 with diurnal warm-layer parameterization.

    Wraps ``coare36`` with a time-sequential warm-layer model (Fairall et al.
    1996) that integrates the heat budget through the diurnal warm layer.  The
    warm-layer depth and temperature anomaly are updated at every time step; the
    corrected near-surface temperature is then passed to ``coare36`` to compute
    the final fluxes.

    This function requires a time-ordered input array.  The time step is inferred
    from successive Julian days.

    Parameters
    ----------
    julian_day : np.ndarray
        Julian day for each record (fractional part = UTC time).
    u, z_u, t, z_t, rh, z_rh, p, ts, sw_down, lw_down, lat, lon,
    pbl_height, rain : array_like
        Same as ``coare36``; see that function for units.
    ts_depth : np.ndarray
        Depth of the sea surface temperature sensor (m), positive downward.
    surface_salinity : np.ndarray, optional
        Sea surface salinity (PSU). Default SSO.
    phase_speed, h_sig : np.ndarray, optional
        Wave parameters (see ``coare36``).
    zref_u, zref_t, zref_rh : float, optional
        Reference heights for profile outputs.

    Returns
    -------
    Dict[str, np.ndarray]
        All outputs from ``coare36`` plus:

        ``dT_warm``
            Temperature anomaly across the full warm layer (C).
        ``dz_warm``
            Warm layer depth (m).
        ``dT_warm_to_skin``
            Temperature anomaly between sensor depth and skin (C).
            Use ``T_skin = ts + dT_warm_to_skin - dT_skin`` to get skin T.
        ``du_warm``
            Current accumulation in the warm layer (m/s).
    """
    julian_day = np.atleast_1d(np.asarray(julian_day, dtype=float)).ravel()
    u = np.atleast_1d(np.asarray(u, dtype=float)).ravel()
    N = u.size

    if surface_salinity is None:
        surface_salinity = np.full(N, SSO)

    def _arr(x):
        return np.broadcast_to(np.atleast_1d(np.asarray(x, dtype=float)), (N,)).copy()

    t = _arr(t)
    rh = _arr(rh)
    p = _arr(p)
    ts = _arr(ts)
    sw_down = _arr(sw_down)
    lw_down = _arr(lw_down)
    lat = _arr(lat)
    lon = _arr(lon)
    pbl_height = _arr(pbl_height)
    rain = _arr(rain)
    ts_depth = _arr(ts_depth)
    surface_salinity = _arr(surface_salinity)
    z_u = _arr(z_u)
    z_t = _arr(z_t)
    z_rh = _arr(z_rh)

    if phase_speed is None:
        phase_speed = np.full(N, np.nan)
    else:
        phase_speed = _arr(phase_speed)
    if h_sig is None:
        h_sig = np.full(N, np.nan)
    else:
        h_sig = _arr(h_sig)

    # Constants for warm-layer model
    cp_water = 4000.0
    rho_water = 1022.0
    rich = 0.65  # critical Richardson number
    max_warm_depth = 19.0  # maximum warm layer depth [m]

    # Initialise warm-layer state
    qcol_ac = 0.0  # accumulated net heat [J/m2]
    tau_ac = 0.0  # accumulated momentum [N s/m2]
    dT_warm = 0.0  # warm-layer temperature anomaly [C]
    du_warm = 0.0  # warm-layer current anomaly [m/s]
    dz_warm = max_warm_depth
    dT_warm_to_skin = 0.0
    fxp = 0.5  # fraction of sw absorbed in warm layer

    jtime = 0.0
    jamset = 0  # flag: warming event has started
    jump = 1  # flag: before first local sunrise

    # Store first time-step fluxes for loop start
    flux0 = coare36(
        u[0],
        z_u[0],
        t[0],
        z_t[0],
        rh[0],
        z_rh[0],
        p[0],
        ts[0],
        sw_down[0],
        lw_down[0],
        julian_day[0],
        lat[0],
        lon[0],
        pbl_height[0],
        rain[0],
        surface_salinity[0],
        phase_speed[0],
        h_sig[0],
        zref_u=zref_u,
        zref_t=zref_t,
        zref_rh=zref_rh,
    )
    tau_prev = flux0["tau"][0]
    hs_prev = flux0["hs"][0]
    hl_prev = flux0["hl"][0]
    dT_skin_prev = flux0["dT_skin"][0]
    hrain_prev = flux0["hrain"][0]

    warm_output = np.full((N, 4), np.nan)  # [dT_warm, dz_warm, dT_warm_to_skin, du_warm]

    for i in range(N):
        jd = julian_day[i]
        sw = sw_down[i]
        lw = lw_down[i]
        tsea = ts[i]
        ss = surface_salinity[i]
        g_i = st.gravity_at_lat(lat[i])

        # Solar absorption parameters for warm layer
        Al = 2.1e-5 * (tsea + 3.2) ** 0.79
        ctd1 = np.sqrt(2 * rich * cp_water / (Al * g_i * rho_water))
        ctd2 = np.sqrt(2 * Al * g_i / (rich * rho_water)) / cp_water**1.5

        # Albedo and sw_net at this time step
        alb_i, _, _, _ = sea_surface_albedo(sw, jd, lat[i], lon[i])
        sw_net_i = (1 - alb_i[0]) * sw
        lw_net_i = 0.97 * (SB * (tsea - dT_skin_prev + T0) ** 4 - lw)

        # Local time: determine if we are in the warming window (after local dawn)
        local_hour = (lon[i] + 7.5) / 15 + (jd - np.fix(jd)) * 24
        local_hour_mod = local_hour % 24
        newtime = local_hour_mod * 3600

        if i > 0:
            if newtime <= 21600 or jump == 0:
                jump = 0
                if newtime < jtime:
                    # New day: reset accumulations
                    jamset = 0
                    fxp = 0.5
                    dz_warm = max_warm_depth
                    tau_ac = 0.0
                    qcol_ac = 0.0
                    dT_warm = 0.0
                    du_warm = 0.0
                else:
                    dtime = newtime - jtime
                    qr_out = lw_net_i + hs_prev + hl_prev + hrain_prev
                    q_pwp = fxp * sw_net_i - qr_out

                    if q_pwp >= 50 or jamset == 1:
                        jamset = 1
                        tau_ac += max(0.002, tau_prev) * dtime

                        if qcol_ac + q_pwp * dtime > 0:
                            # Iterate absorption profile 5 times
                            for _ in range(5):
                                fxp = (
                                    1
                                    - (
                                        0.28 * 0.014 * (1 - np.exp(-dz_warm / 0.014))
                                        + 0.27 * 0.357 * (1 - np.exp(-dz_warm / 0.357))
                                        + 0.45 * 12.82 * (1 - np.exp(-dz_warm / 12.82))
                                    )
                                    / dz_warm
                                )
                                qjoule = (fxp * sw_net_i - qr_out) * dtime
                                if qcol_ac + qjoule > 0:
                                    dz_warm = min(max_warm_depth, ctd1 * tau_ac / np.sqrt(qcol_ac + qjoule))
                        else:
                            fxp = 0.75
                            dz_warm = max_warm_depth
                            qjoule = (fxp * sw_net_i - qr_out) * dtime

                        qcol_ac += qjoule

                        if qcol_ac > 0:
                            dT_warm = ctd2 * qcol_ac**1.5 / tau_ac
                            du_warm = 2 * tau_ac / (dz_warm * rho_water)
                        else:
                            dT_warm = 0.0
                            du_warm = 0.0

            # Warm-layer contribution to skin temperature
            if dz_warm < ts_depth[i]:
                dT_warm_to_skin = dT_warm
            else:
                dT_warm_to_skin = dT_warm * ts_depth[i] / dz_warm

        jtime = newtime

        warm_output[i, 0] = dT_warm
        warm_output[i, 1] = dz_warm
        warm_output[i, 2] = dT_warm_to_skin
        warm_output[i, 3] = du_warm

        # Run coare36 at this step to get fluxes for next step
        ts_adj = tsea + dT_warm_to_skin
        fl = coare36(
            u[i],
            z_u[i],
            t[i],
            z_t[i],
            rh[i],
            z_rh[i],
            p[i],
            ts_adj,
            sw,
            lw,
            jd,
            lat[i],
            lon[i],
            pbl_height[i],
            rain[i],
            ss,
            phase_speed[i],
            h_sig[i],
            zref_u=zref_u,
            zref_t=zref_t,
            zref_rh=zref_rh,
        )
        tau_prev = fl["tau"][0]
        hs_prev = fl["hs"][0]
        hl_prev = fl["hl"][0]
        dT_skin_prev = fl["dT_skin"][0]
        hrain_prev = fl["hrain"][0]

    # NaN-fill output for bad solar inputs
    bad = np.isnan(sw_down)
    warm_output[bad, :] = np.nan

    # Final: rerun coare36 on entire time series with warm-layer-corrected SST
    ts_corrected = ts + warm_output[:, 2]
    result = coare36(
        u,
        z_u,
        t,
        z_t,
        rh,
        z_rh,
        p,
        ts_corrected,
        sw_down,
        lw_down,
        julian_day,
        lat,
        lon,
        pbl_height,
        rain,
        surface_salinity,
        phase_speed,
        h_sig,
        zref_u=zref_u,
        zref_t=zref_t,
        zref_rh=zref_rh,
    )

    result["dT_warm"] = warm_output[:, 0]
    result["dz_warm"] = warm_output[:, 1]
    result["dT_warm_to_skin"] = warm_output[:, 2]
    result["du_warm"] = warm_output[:, 3]
    return result
