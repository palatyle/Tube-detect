from datetime import datetime

import data_wrangle as dw
import numpy as np

# Inputs for thermal inertia calculation
input_day_thermal = ("/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/data/thermal.tif")
input_night_thermal = "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/data/HHA_night_hres_align_clip.tif"
input_albedo = "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/data/Albedo_hres_align_clip.tif"

# Read in rasters
raster_day, raster_GDAL_day, _ = dw.read_in_raster(input_day_thermal)
raster_night, raster_GDAL_night, _ = dw.read_in_raster(input_night_thermal)
albedo, albedo_GDAL, _ = dw.read_in_raster(input_albedo)

# Constants
angular_frequency = 7.27 * 10 ** (-5)  # rad/sec                maltese et al
day_satellite_overpass_time = float(1121 * 60 * 60)  # seconds since midnight
night_satellite_overpass_time = float(246 * 60 * 60)  # seconds since midnight
max_temp_time = float(1400 * 60 * 60)  # seconds since midnight    scheidt et al
min_temp_time = float(200 * 60 * 60)  # seconds since midnight     scheidt et al
latitude = 43.4956
solar_constant = 1367  # W/m**2 for earth        V. M. Fedorov
Ct_transmittance = 0.75  # atmospheric transmittance for Earth     scheidt et al
day_temp = raster_day
night_temp = raster_night
flight_dt = datetime.fromisoformat("2022-06-16 11:21:00")  # Get timezone aware datetime object
flight_dt_tz_aware = dw.get_dt_obj(flight_dt, "US/Mountain")


def calc_max_temp(d_temp, n_temp, max_t_time, d_sat_time, ang_freq, n_sat_time):
    """Calculates maximum temperature in diurnal cycle based on given variables.

    Args:
        d_temp (arr): daytime temperature
        n_temp (arr): nighttime temperature
        max_t_time (int): time of maximum temperature
        d_sat_time (int): time of satellite overpass day
        ang_freq (float): rotational angular frequency of earth
        n_sat_time (int): time of satellite overpass night

    Returns:
        max_temp (masked_array) : maximum temperature
    """

    max_temp = d_temp + (
        ((d_temp - n_temp)
            * (np.cos(ang_freq * max_t_time) - np.cos(ang_freq * d_sat_time))
        )
        / (np.cos(ang_freq * d_sat_time) - np.cos(ang_freq * n_sat_time))
    )

    return max_temp


def calc_min_temp(n_temp, d_temp, ang_freq, min_t_time, n_sat_time, d_sat_time):
    """Calculates minimum temperature in diurnal cycle based on given variables.

    Args:
        n_temp (arr): nighttime temperature
        d_temp (arr): daytime temperature
        ang_freq (float): rotational angular frequency of earth
        min_t_time (int): time of minimum temperature
        n_sat_time (int): time of satellite overpass night
        d_sat_time (int): time of satellite overpass day

    Returns:
        min_temp (masked_array) : minimum temperature
    """
    min_temp = n_temp + (
        ((d_temp - n_temp)
            * (np.cos(ang_freq * min_t_time) - np.cos(ang_freq * n_sat_time))
        )
        / (np.cos(ang_freq * d_sat_time) - np.cos(ang_freq * n_sat_time))
    )

    return min_temp


def calc_temp_range(max_T, min_T):
    """Calculates the temperature range over a dirunal cycle.

    Args:
        max_T (masked_array): Maximum temperature
        min_T (masked_array): Minimum Temperature

    Returns:
        t_range (masked_array): range of temperatures
    """
    t_range = max_T - min_T
    return t_range


def calc_temp_ratio(max_T, min_T):
    """Calculates a temperature ratio over a diurnal cycle.

    Args:
        max_T (masked_array): Maximum Temperature
        min_T (masked_array): Minimum Temperature

    Returns:
        t_ratio (masked_array) : ratio of temperatures
    """
    t_ratio = max_T / min_T
    return t_ratio


def calc_ATI(alb, t_range):
    """Calculates Apparent Thermal Inertia over a diurnal cycle.

    Args:
        alb (masked_array): Previously calculated albedo for study area
        t_range (masked_array): range of temperatures

    Returns:
        ATI (masked_array): Apparent Thermal Inertia for study area
    """
    ATI = (1 - alb) / t_range
    return ATI


def calc_b(ang_freq, max_t_time):
    """Calculates the b parameter.

    Args:
        ang_freq (float): rotational angular frequency of earth
        max_t_time (int): time of maximum temperature

    Returns:
        b_var (float): b parameter
    """
    b_var = (np.tan(ang_freq * max_t_time)) / (1 - (np.tan(ang_freq * max_t_time)))
    return b_var


def calc_solar_dec(flight_dt_tz):
    """Calculates Solar Declination based on given variable.

    Args:
        flight_dt_tz (dt): timezone aware date time object

    Returns:
        solar_dec (float): solar declination
    """
    solar_dec = np.deg2rad(dw.calc_sol_dec(flight_dt_tz))
    return solar_dec


def calc_pd_1(b_var):
    """Calculates the first phase difference variable. Scheidt et al.

    Args:
        b_var (float): b variable

    Returns:
        pd_1 (float): phase difference 1
    """
    pd_1 = np.arctan(b_var / (1 + b_var))
    return pd_1


def calc_pd_2(b_var):
    """Calculates the second phase difference variable. Scheidt et al.

    Args:
        b_var (float): b variable

    Returns:
        pd_2 (float): phase difference 2
    """
    pd_2 = np.arctan((b_var * np.sqrt(2)) / (1 + (b_var * np.sqrt(2))))
    return pd_2


def calc_xi(solar_dec, lat):
    """Calculates the xi constant based on given variables.

    Args:
        solar_dec (float): solar declination
        lat (float): latitude of study area

    Returns:
        xi (float) : xi constant
    """
    xi = np.arccos(np.tan(solar_dec) * np.tan(np.deg2rad(lat)))
    return xi


def calc_A1(solar_dec, lat, xi):
    """Calculates the A1 Fourier coefficient based on given variables.

    Args:
        solar_dec (float): solar declination
        lat (float): latitude of study area
        xi (float): xi constant

    Returns:
        A1 (float) : A1 fourier coefficient
    """
    A1 = ((2 / np.pi) * (np.sin(solar_dec) * (np.sin(np.deg2rad(lat))))) + (
        (1 / 2 * np.pi) * (np.cos(solar_dec) * (np.cos(np.deg2rad(lat))))
    ) * ((np.sin(2 * xi) + (2 * xi)))
    return A1


def calc_A2(solar_dec,lat,xi):
    """Calculates the A2 Fourier coefficient based on given variables.

    Args:
        solar_dec (float): solar declination
        lat (float): latitude of study area
        xi (float): xi constant

    Returns:
        A2 (float) : A2 Fourier coefficient
    """
    A2 = (
        ((2 * np.sin(solar_dec) * (np.sin(np.deg2rad(lat)))) / (2 * np.pi))
        * (np.sin(2 * xi))
    ) + (
        (2 * np.cos(solar_dec) * (np.cos(np.deg2rad(lat))) / (np.pi * (2**2 - 1)))
        * ((2 * (np.sin(2 * xi)) * (np.cos(xi))) - ((np.cos(2 * xi)) * (np.sin(xi))))
    )
    return A2


def calc_TI(ATI, sol_const, Ct, ang_freq, A1, night_sat_time, pd1, day_sat_time, b_var, A2, pd2):
    """Calculates thermal inertia based on given variables.

    Args:
        ATI (arr): ATI for study area
        sol_const (int): solar constant for Earth
        Ct (int): CT Transmittance for Earth
        ang_freq (float): rotational angular frequency of Earth
        A1 (float): A1 fourier coefficient
        night_sat_time (int): time of satellite overpass night
        pd1 (float): phase difference 1
        day_sat_time (int): time of satellite overpass day
        b_var (float): b variable
        A2 (float): A2 fourier coefficient
        pd2 (float): phase difference 2

    Returns:
        arr : Thermal Inertia for study area
    """
    TI = (ATI * ((sol_const * Ct) / np.sqrt(ang_freq))) * (
        (
            (
                A1
                * (
                    (np.cos((ang_freq * night_sat_time) - pd1))
                    - (np.cos((ang_freq * day_sat_time) - pd1))
                )
            )
            / (np.sqrt(1 + (1 / b_var) + (1 / (2 * b_var**2))))
        )
        + (
            (
                A2
                + (
                    (np.cos((ang_freq * night_sat_time) - pd2))
                    - (np.cos((ang_freq * day_sat_time) - pd2))
                )
            )
            / (np.sqrt(2 + (np.sqrt(2) / b_var) + (1 / (2 * b_var**2))))
        )
    )
    return TI


# Calculate max temperature
max_Temperature = calc_max_temp(
    day_temp / 100.0,
    night_temp / 100.0,
    max_temp_time,
    day_satellite_overpass_time,
    angular_frequency,
    night_satellite_overpass_time,
)

# Calculate min temperature
min_Temperature = calc_min_temp(
    night_temp / 100.0,
    day_temp / 100.0,
    angular_frequency,
    min_temp_time,
    night_satellite_overpass_time,
    day_satellite_overpass_time,
)

# Calculate temperature range between max and min. Divide by 100 to get from centikelvin to kelvin
temp_range = calc_temp_range(max_Temperature, night_temp / 100.0)

# Calculate temperaure ratio between max and min. Divide by 100 to get from centikelvin to kelvin
temp_ratio = calc_temp_ratio(max_Temperature, night_temp / 100.0)

# Calculate apparent thermal inertia
apparent_thermal_inertia = calc_ATI(albedo, temp_range)

# Calculate b parameter
b = calc_b(angular_frequency, max_temp_time)

# Calculate solar declination
solar_declination = calc_solar_dec(flight_dt_tz_aware)

# Calculate phase difference 1
phase_diff_1 = calc_pd_1(b)

# Calculate phase difference 2
phase_diff_2 = calc_pd_2(b)

# Calculate xi constant
xi_constant = calc_xi(solar_declination, latitude)

# Calculate A1 fourier coefficient
A1_fourier = calc_A1(solar_declination, latitude, xi_constant)

# Calculate A2 fourier coefficient
A2_fourier = calc_A2(solar_declination, latitude, xi_constant)

# Calculate thermal inertia using max temp time instead of overpass time.
thermal_inertia = calc_TI(
    apparent_thermal_inertia,
    solar_constant,
    Ct_transmittance,
    angular_frequency,
    A1_fourier,
    night_satellite_overpass_time,
    phase_diff_1,
    max_temp_time,
    b,
    A2_fourier,
    phase_diff_2,
)


print(np.median(max_Temperature))
print(np.median(night_temp / 100.0))


# converting array to tiff file
dw.write_band(
    albedo_GDAL,
    thermal_inertia,
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/data",
    "thermal_inertia.tiff",
    None,
)

dw.write_band(
    albedo_GDAL,
    temp_ratio,
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/data/",
    "temperature_ratio.tiff",
    None,
)

dw.write_band(
    albedo_GDAL,
    max_Temperature,
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/data/",
    "Max_temp_Scheidt.tiff",
    None,
)
dw.write_band(
    albedo_GDAL,
    night_temp / 100.0,
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/data/",
    "min_temp_kelvin.tiff",
    None,
)

dw.write_band(
    albedo_GDAL,
    apparent_thermal_inertia,
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/data/",
    "ATI.tiff",
    None,
)
