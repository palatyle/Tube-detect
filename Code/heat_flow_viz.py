#%% imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sec2day(time_df):
    """Convert seconds to days.

    Parameters
    ----------
    time_df : dataframe
        Dataframe containing time in seconds.

    Returns
    -------
    dataframe
        Dataframe containing time in days.
    """    
    return time_df / 60 / 60 / 24


def sec2week(time_df):
    """Convert seconds to weeks.

    Parameters
    ----------
    time_df : dataframe
        Dataframe containing time in seconds.

    Returns
    -------
    dataframe
        Dataframe containing time in weeks.
    """    
    return time_df / 60 / 60 / 24 / 7


def sec2year(time_df):
    """Convert seconds to years.

    Parameters
    ----------
    time_df : dataframe
        Dataframe containing time in seconds.

    Returns
    -------
    dataframe
        Dataframe containing time in years.
    """    
    return time_df / 60 / 60 / 24 / 365


def find_peaks(df, min, max):
    """Find index of peak values in a dataframe between min and max times

    Parameters
    ----------
    df : dataframe
        Dataframe containing time and surf_temp columns.
    min : float/int
        Minimum time to search for peaks.
    max : float/int
        Maximum time to search for peaks.

    Returns
    -------
    list
        List of indices of peak values.
    """    
    range_idx = np.where(np.logical_and(df.time >= float(min), df.time <= max))
    return np.max(df.surf_temp[range_idx[0]])


# Plotting parameters
plt.rcParams.update({"font.sans-serif": "Myriad Pro"})
plt.rcParams.update(
    {
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.titlesize": 22,
        "figure.titlesize": 16,
        "axes.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 14,
        "legend.title_fontsize": 14,
        "figure.facecolor": (240 / 255, 240 / 255, 240 / 255),
        "savefig.facecolor": (240 / 255, 240 / 255, 240 / 255),
    }
)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

#%% Earth
Earth_300K = pd.read_csv(
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/heat_flow_reg-0m_lower_bound-300_K.csv"
)
Earth_273K = pd.read_csv(
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/heat_flow_reg-0m_lower_bound-273_K.csv"
)

fig, ax = plt.subplots()
Earth_300K.time = sec2day(Earth_300K.time)
Earth_273K.time = sec2day(Earth_273K.time)
Earth_300K.plot(x="time", y="surf_temp-300", kind="line", ax=ax, label="300 K")
Earth_273K.plot(x="time", y="surf_temp-273", kind="line", ax=ax, label="273 K")

ax.set_xlim(300, 301)
ax.set_ylabel("Surface Temperature [K]")
ax.set_xlabel("Days")
ax.set_title("Earth - 5m roof")

ax.legend(title="Lower Boundary Temperature")

fig.savefig("Earth_tube_vs_no_tube.pdf", format="pdf", bbox_inches=None, dpi=300)

#%% Earth 3 m
Earth_300K_3m = pd.read_csv(
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/heat_flow_tube_roof_Earth_3.0m_reg-0m_lower_bound-300_K.csv"
)
Earth_273K_3m = pd.read_csv(
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/heat_flow_tube_roof_Earth_3.0m_reg-0m_lower_bound-273.0_K.csv"
)

fig_3m, ax_3m = plt.subplots()
Earth_300K_3m.time = sec2day(Earth_300K_3m.time)
Earth_273K_3m.time = sec2day(Earth_273K_3m.time)
Earth_300K_3m.plot(x="time", y="surf_temp", kind="line", ax=ax_3m, label="300 K")
Earth_273K_3m.plot(x="time", y="surf_temp", kind="line", ax=ax_3m, label="273 K")

ax_3m.set_xlim(300, 301)
ax_3m.set_ylabel("Surface Temperature [K]")
ax_3m.set_xlabel("Days")
ax_3m.set_title("Earth - 3m roof")

ax_3m.legend(title="Lower Boundary Temperature")

fig_3m.savefig("Earth_tube_vs_no_tube_3m.pdf", format="pdf", bbox_inches=None, dpi=300)

#%% Earth 1 m
Earth_300K_1m = pd.read_csv(
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/heat_flow_tube_roof_Earth_1.0m_reg-0m_lower_bound-300_K.csv"
)
Earth_273K_1m = pd.read_csv(
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/heat_flow_tube_roof_Earth_1.0m_reg-0m_lower_bound-273.0_K.csv"
)

fig_1m, ax_1m = plt.subplots()
Earth_300K_1m.time = sec2day(Earth_300K_1m.time)
Earth_273K_1m.time = sec2day(Earth_273K_1m.time)
Earth_300K_1m.plot(x="time", y="surf_temp", kind="line", ax=ax_1m, label="300 K")
Earth_273K_1m.plot(x="time", y="surf_temp", kind="line", ax=ax_1m, label="273 K")

ax_1m.set_xlim(300, 301)
ax_1m.set_ylabel("Surface Temperature [K]")
ax_1m.set_xlabel("Days")
ax_1m.set_title("Earth - 1m roof")

ax_1m.legend(title="Lower Boundary Temperature")

fig_1m.savefig("Earth_tube_vs_no_tube_1m.pdf", format="pdf", bbox_inches=None, dpi=300)

#%% Earth all tube roofs

fig_all, ax_all = plt.subplots(1, 3, sharey=True)

Earth_300K.plot(x="time", y="surf_temp-300", kind="line", ax=ax_all[0], label="300 K")
Earth_273K.plot(x="time", y="surf_temp-273", kind="line", ax=ax_all[0], label="273 K")

ax_all[0].set_xlim(300, 301)
ax_all[0].set_ylabel("Surface Temperature [K]")
ax_all[0].set_xlabel("Days")
ax_all[0].set_title("5m roof")

ax_all[0].legend(title="Lower Boundary Temperature")

Earth_300K_3m.plot(
    x="time", y="surf_temp", kind="line", ax=ax_all[1], legend=False, label="_nolegend_"
)
Earth_273K_3m.plot(
    x="time", y="surf_temp", kind="line", ax=ax_all[1], legend=False, label="_nolegend_"
)

ax_all[1].set_xlim(300, 301)
ax_all[1].set_ylabel("Surface Temperature [K]")
ax_all[1].set_xlabel("Days")
ax_all[1].set_title("3m roof")

Earth_300K_1m.plot(
    x="time", y="surf_temp", kind="line", ax=ax_all[2], legend=False, label="_nolegend_"
)
Earth_273K_1m.plot(
    x="time", y="surf_temp", kind="line", ax=ax_all[2], legend=False, label="_nolegend_"
)

ax_all[2].set_xlim(300, 301)
ax_all[2].set_ylabel("Surface Temperature [K]")
ax_all[2].set_xlabel("Days")
ax_all[2].set_title("1m roof")

fig_all.savefig(
    "Earth_tube_vs_no_tube_all.pdf", format="pdf", bbox_inches=None, dpi=300
)

#%% Moon

# 290 K lower bound
Moon_reg_0_lb_290 = pd.read_csv(
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/lunar_final_models/heat_flow_tube_roof_Moon_5.0m_reg-0.0m_lower_bound-290.0_K.csv"
)
Moon_reg_0_1_lb_290 = pd.read_csv(
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/lunar_final_models/heat_flow_tube_roof_Moon_5.0m_reg-0.1m_lower_bound-290.0_K.csv"
)
Moon_reg_0_5_lb_290 = pd.read_csv(
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/lunar_final_models/heat_flow_tube_roof_Moon_5.0m_reg-0.5m_lower_bound-290.0_K.csv"
)
Moon_reg_1_lb_290 = pd.read_csv(
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/lunar_final_models/heat_flow_tube_roof_Moon_5.0m_reg-1.0m_lower_bound-290.0_K.csv"
)
Moon_reg_5_lb_290 = pd.read_csv(
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/lunar_final_models/heat_flow_tube_roof_Moon_5.0m_reg-5.0m_lower_bound-290.0_K.csv"
)

# 45 K lower bound
Moon_reg_0_lb_45 = pd.read_csv(
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/lunar_final_models/heat_flow_tube_roof_Moon_5.0m_reg-0.0m_lower_bound-45.0_K.csv"
)
Moon_reg_0_1_lb_45 = pd.read_csv(
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/lunar_final_models/heat_flow_tube_roof_Moon_5.0m_reg-0.1m_lower_bound-45.0_K.csv"
)
Moon_reg_0_5_lb_45 = pd.read_csv(
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/lunar_final_models/heat_flow_tube_roof_Moon_5.0m_reg-0.5m_lower_bound-45.0_K.csv"
)
Moon_reg_1_lb_45 = pd.read_csv(
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/lunar_final_models/heat_flow_tube_roof_Moon_5.0m_reg-1.0m_lower_bound-45.0_K.csv"
)
Moon_reg_5_lb_45 = pd.read_csv(
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/lunar_final_models/heat_flow_tube_roof_Moon_5.0m_reg-5.0m_lower_bound-45.0_K.csv"
)

# Convert seconds
Moon_reg_0_lb_290.time = sec2year(Moon_reg_0_lb_290.time)
Moon_reg_0_1_lb_290.time = sec2year(Moon_reg_0_1_lb_290.time)
Moon_reg_0_5_lb_290.time = sec2year(Moon_reg_0_5_lb_290.time)
Moon_reg_1_lb_290.time = sec2year(Moon_reg_1_lb_290.time)
Moon_reg_5_lb_290.time = sec2year(Moon_reg_5_lb_290.time)

Moon_reg_0_lb_45.time = sec2year(Moon_reg_0_lb_45.time)
Moon_reg_0_1_lb_45.time = sec2year(Moon_reg_0_1_lb_45.time)
Moon_reg_0_5_lb_45.time = sec2year(Moon_reg_0_5_lb_45.time)
Moon_reg_1_lb_45.time = sec2year(Moon_reg_1_lb_45.time)
Moon_reg_5_lb_45.time = sec2year(Moon_reg_5_lb_45.time)

fig1, ax1 = plt.subplots(1, 2, sharey=True)

colors = plt.cm.viridis(np.linspace(0, 1, num=5))

Moon_reg_0_lb_290.plot(
    x="time",
    y="surf_temp",
    kind="line",
    ax=ax1[0],
    legend=False,
    label="0 m",
    color=colors[0],
)
Moon_reg_0_1_lb_290.plot(
    x="time",
    y="surf_temp",
    kind="line",
    ax=ax1[0],
    legend=False,
    label="0.1 m",
    color=colors[1],
)
Moon_reg_0_5_lb_290.plot(
    x="time",
    y="surf_temp",
    kind="line",
    ax=ax1[0],
    legend=False,
    label="0.5 m",
    color=colors[2],
)
Moon_reg_1_lb_290.plot(
    x="time",
    y="surf_temp",
    kind="line",
    ax=ax1[0],
    legend=False,
    label="1 m",
    color=colors[3],
)
Moon_reg_5_lb_290.plot(
    x="time",
    y="surf_temp",
    kind="line",
    ax=ax1[0],
    legend=False,
    label="5 m",
    color=colors[4],
)


ax1[0].set_xlim(70.05, 70.16)
ax1[0].set_ylabel("Surface Temperature [K]")
ax1[0].set_xlabel("Years")
ax1[0].set_title("290 K lower Boundary")

Moon_reg_0_lb_45.plot(
    x="time",
    y="surf_temp",
    kind="line",
    ax=ax1[1],
    legend=False,
    label="_nolegend_",
    color=colors[0],
)
Moon_reg_0_1_lb_45.plot(
    x="time",
    y="surf_temp",
    kind="line",
    ax=ax1[1],
    legend=False,
    label="_nolegend_",
    color=colors[1],
)
Moon_reg_0_5_lb_45.plot(
    x="time",
    y="surf_temp",
    kind="line",
    ax=ax1[1],
    legend=False,
    label="_nolegend_",
    color=colors[2],
)
Moon_reg_1_lb_45.plot(
    x="time",
    y="surf_temp",
    kind="line",
    ax=ax1[1],
    legend=False,
    label="_nolegend_",
    color=colors[3],
)
Moon_reg_5_lb_45.plot(
    x="time",
    y="surf_temp",
    kind="line",
    ax=ax1[1],
    legend=False,
    label="_nolegend_",
    color=colors[4],
)


ax1[1].set_xlim(70.05, 70.16)
# ax1[1].set_ylabel("Surface Temperature [K]")
ax1[1].set_xlabel("Years")
ax1[1].set_title("45 K lower Boundary")


fig1.legend(title="Regolith Depth")
fig1.savefig("Moon_reg_depth.pdf", format="pdf", bbox_inches=None, dpi=300)


fig2, ax2 = plt.subplots(1, 3, sharey=True)

Moon_reg_0_lb_290.plot(
    x="time",
    y="surf_temp",
    kind="line",
    ax=ax2[0],
    legend=False,
    label="290 K"
)

Moon_reg_0_lb_45.plot(
    x="time",
    y="surf_temp",
    kind="line",
    ax=ax2[0],
    legend=False,
    label="45 K"
)
ax2[0].set_xlim(70, 70.08)
ax2[0].set_xlabel("Years")
ax2[0].set_ylabel("Surface Temperature [K]")
ax2[0].set_title("0 m regolith")

Moon_reg_1_lb_290.plot(
    x="time",
    y="surf_temp",
    kind="line",
    ax=ax2[1],
    legend=False,
    label="_nolegend_"
)
Moon_reg_1_lb_45.plot(
    x="time",
    y="surf_temp",
    kind="line",
    ax=ax2[1],
    legend=False,
    label="_nolegend_"
)
ax2[1].set_xlim(70, 70.08)
ax2[1].set_xlabel("Years")
ax2[1].set_title("1 m regolith")

Moon_reg_5_lb_290.plot(
    x="time", 
    y="surf_temp", 
    kind="line", 
    ax=ax2[2], 
    legend=False, 
    label="_nolegend_"
)
Moon_reg_5_lb_45.plot(
    x="time", 
    y="surf_temp", 
    kind="line", 
    ax=ax2[2], 
    legend=False, 
    label="_nolegend_"
)
ax2[2].set_xlim(70, 70.08)
ax2[2].set_xlabel("Years")
ax2[2].set_title("5 m regolith")

fig2.legend(title="Lower Boundary Temperature")
fig2.savefig("Moon_lower_bounds.pdf", format="pdf", bbox_inches=None, dpi=300)

# %%
Moon_290 = []
Moon_45 = []
reg_depths = [5, 1, 0.5, 0.1, 0]


Moon_290.append(find_peaks(Moon_reg_5_lb_290, 70, 70.08))
Moon_290.append(find_peaks(Moon_reg_1_lb_290, 70, 70.08))
Moon_290.append(find_peaks(Moon_reg_0_5_lb_290, 70, 70.08))
Moon_290.append(find_peaks(Moon_reg_0_1_lb_290, 70, 70.08))
Moon_290.append(find_peaks(Moon_reg_0_lb_290, 70, 70.08))

Moon_45.append(find_peaks(Moon_reg_5_lb_45, 70, 70.08))
Moon_45.append(find_peaks(Moon_reg_1_lb_45, 70, 70.08))
Moon_45.append(find_peaks(Moon_reg_0_5_lb_45, 70, 70.08))
Moon_45.append(find_peaks(Moon_reg_0_1_lb_45, 70, 70.08))
Moon_45.append(find_peaks(Moon_reg_0_lb_45, 70, 70.08))

Moon_diff = np.subtract(Moon_290, Moon_45)

fig_diff, ax_diff = plt.subplots()

ax_diff.plot(reg_depths, Moon_diff, "o-")
ax_diff.set_xlabel("Regolith Depth [m]")
ax_diff.set_ylabel("Temperature Difference [m]")
ax_diff.set_title("Temperature Difference Between 290 K and 45 K Runs")

fig_diff.savefig("Moon_290vs45_diff.pdf", format="pdf", bbox_inches=None, dpi=300)


print(Moon_diff)
print(np.mean(Moon_diff))
print("stop")

# %% Moon 100 m roof

# 290 K lower bound
Moon_reg_0_lb_290_tr_100 = pd.read_csv(
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/lunar_final_models/heat_flow_tube_roof_Moon_100.0m_reg-0.0m_lower_bound-290.0_K.csv"
)
Moon_reg_0_1_lb_290_tr_100 = pd.read_csv(
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/lunar_final_models/heat_flow_tube_roof_Moon_100.0m_reg-0.1m_lower_bound-290.0_K.csv"
)
Moon_reg_0_5_lb_290_tr_100 = pd.read_csv(
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/lunar_final_models/heat_flow_tube_roof_Moon_100.0m_reg-0.5m_lower_bound-290.0_K.csv"
)
Moon_reg_1_lb_290_tr_100 = pd.read_csv(
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/lunar_final_models/heat_flow_tube_roof_Moon_100.0m_reg-1.0m_lower_bound-290.0_K.csv"
)
Moon_reg_5_lb_290_tr_100 = pd.read_csv(
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/lunar_final_models/heat_flow_tube_roof_Moon_100.0m_reg-5.0m_lower_bound-290.0_K.csv"
)

# 45 K lower bound
Moon_reg_0_lb_45_tr_100 = pd.read_csv(
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/lunar_final_models/heat_flow_tube_roof_Moon_100.0m_reg-0.0m_lower_bound-45.0_K.csv"
)
Moon_reg_0_1_lb_45_tr_100 = pd.read_csv(
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/lunar_final_models/heat_flow_tube_roof_Moon_100.0m_reg-0.1m_lower_bound-45.0_K.csv"
)
Moon_reg_0_5_lb_45_tr_100 = pd.read_csv(
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/lunar_final_models/heat_flow_tube_roof_Moon_100.0m_reg-0.5m_lower_bound-45.0_K.csv"
)
Moon_reg_1_lb_45_tr_100 = pd.read_csv(
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/lunar_final_models/heat_flow_tube_roof_Moon_100.0m_reg-1.0m_lower_bound-45.0_K.csv"
)
Moon_reg_5_lb_45_tr_100 = pd.read_csv(
    "/Users/tylerpaladino/Documents/ISU/Thesis/Lava_tube_detection/lunar_final_models/heat_flow_tube_roof_Moon_100.0m_reg-5.0m_lower_bound-45.0_K.csv"
)

# Convert seconds
Moon_reg_0_lb_290_tr_100.time = sec2year(Moon_reg_0_lb_290_tr_100.time)
Moon_reg_0_1_lb_290_tr_100.time = sec2year(Moon_reg_0_1_lb_290_tr_100.time)
Moon_reg_0_5_lb_290_tr_100.time = sec2year(Moon_reg_0_5_lb_290_tr_100.time)
Moon_reg_1_lb_290_tr_100.time = sec2year(Moon_reg_1_lb_290_tr_100.time)
Moon_reg_5_lb_290_tr_100.time = sec2year(Moon_reg_5_lb_290_tr_100.time)

Moon_reg_0_lb_45_tr_100.time = sec2year(Moon_reg_0_lb_45_tr_100.time)
Moon_reg_0_1_lb_45_tr_100.time = sec2year(Moon_reg_0_1_lb_45_tr_100.time)
Moon_reg_0_5_lb_45_tr_100.time = sec2year(Moon_reg_0_5_lb_45_tr_100.time)
Moon_reg_1_lb_45_tr_100.time = sec2year(Moon_reg_1_lb_45_tr_100.time)
Moon_reg_5_lb_45_tr_100.time = sec2year(Moon_reg_5_lb_45_tr_100.time)

fig1_100, ax1_100 = plt.subplots(1, 2, sharey=True)

colors = plt.cm.viridis(np.linspace(0, 1, num=5))

Moon_reg_0_lb_290_tr_100.plot(
    x="time",
    y="surf_temp",
    kind="line",
    ax=ax1_100[0],
    legend=False,
    label="0 m",
    color=colors[0],
)
Moon_reg_0_1_lb_290_tr_100.plot(
    x="time",
    y="surf_temp",
    kind="line",
    ax=ax1_100[0],
    legend=False,
    label="0.1 m",
    color=colors[1],
)
Moon_reg_0_5_lb_290_tr_100.plot(
    x="time",
    y="surf_temp",
    kind="line",
    ax=ax1_100[0],
    legend=False,
    label="0.5 m",
    color=colors[2],
)
Moon_reg_1_lb_290_tr_100.plot(
    x="time",
    y="surf_temp",
    kind="line",
    ax=ax1_100[0],
    legend=False,
    label="1 m",
    color=colors[3],
)
Moon_reg_5_lb_290_tr_100.plot(
    x="time",
    y="surf_temp",
    kind="line",
    ax=ax1_100[0],
    legend=False,
    label="5 m",
    color=colors[4],
)


ax1_100[0].set_xlim(110.05, 110.16)
ax1_100[0].set_ylabel("Surface Temperature [K]")
ax1_100[0].set_xlabel("Years")
ax1_100[0].set_title("290 K lower Boundary")

Moon_reg_0_lb_45_tr_100.plot(
    x="time",
    y="surf_temp",
    kind="line",
    ax=ax1_100[1],
    legend=False,
    label="_nolegend_",
    color=colors[0],
)
Moon_reg_0_1_lb_45_tr_100.plot(
    x="time",
    y="surf_temp",
    kind="line",
    ax=ax1_100[1],
    legend=False,
    label="_nolegend_",
    color=colors[1],
)
Moon_reg_0_5_lb_45_tr_100.plot(
    x="time",
    y="surf_temp",
    kind="line",
    ax=ax1_100[1],
    legend=False,
    label="_nolegend_",
    color=colors[2],
)
Moon_reg_1_lb_45_tr_100.plot(
    x="time",
    y="surf_temp",
    kind="line",
    ax=ax1_100[1],
    legend=False,
    label="_nolegend_",
    color=colors[3],
)
Moon_reg_5_lb_45_tr_100.plot(
    x="time",
    y="surf_temp",
    kind="line",
    ax=ax1_100[1],
    legend=False,
    label="_nolegend_",
    color=colors[4],
)


ax1_100[1].set_xlim(110.05, 110.16)
# ax1[1].set_ylabel("Surface Temperature [K]")
ax1_100[1].set_xlabel("Years")
ax1_100[1].set_title("45 K lower Boundary")


fig1_100.legend(title="Regolith Depth")
fig1_100.savefig("Moon_reg_depth_tr_100.pdf", format="pdf", bbox_inches=None, dpi=300)


fig2_100, ax2_100 = plt.subplots(1, 3, sharey=True)
Moon_reg_0_lb_290_tr_100.plot(
    x="time", 
    y="surf_temp", 
    kind="line", 
    ax=ax2_100[0], 
    legend=False, 
    label="290 K"
)
Moon_reg_0_lb_45_tr_100.plot(
    x="time", 
    y="surf_temp", 
    kind="line", 
    ax=ax2_100[0], 
    legend=False, 
    label="45 K"
)
ax2_100[0].set_xlim(110, 110.08)
ax2_100[0].set_xlabel("Years")
ax2_100[0].set_ylabel("Surface Temperature [K]")
ax2_100[0].set_title("0 m regolith")

Moon_reg_1_lb_290_tr_100.plot(
    x="time",
    y="surf_temp",
    kind="line",
    ax=ax2_100[1],
    legend=False,
    label="_nolegend_",
)
Moon_reg_1_lb_45_tr_100.plot(
    x="time",
    y="surf_temp",
    kind="line",
    ax=ax2_100[1],
    legend=False,
    label="_nolegend_",
)
ax2_100[1].set_xlim(110, 110.08)
ax2_100[1].set_xlabel("Years")
ax2_100[1].set_title("1 m regolith")

Moon_reg_5_lb_290_tr_100.plot(
    x="time",
    y="surf_temp",
    kind="line",
    ax=ax2_100[2],
    legend=False,
    label="_nolegend_",
)
Moon_reg_5_lb_45_tr_100.plot(
    x="time",
    y="surf_temp",
    kind="line",
    ax=ax2_100[2],
    legend=False,
    label="_nolegend_",
)
ax2_100[2].set_xlim(110, 110.08)
ax2_100[2].set_xlabel("Years")
ax2_100[2].set_title("5 m regolith")

fig2_100.legend(title="Lower Boundary Temperature")
fig2_100.savefig(
    "Moon_lower_bounds_tr_100.pdf", format="pdf", bbox_inches=None, dpi=300
)

# %% 100 m tube roof


Moon_290_tr_100 = []
Moon_45_tr_100 = []
reg_depths = [5, 1, 0.5, 0.1, 0]


Moon_290_tr_100.append(find_peaks(Moon_reg_5_lb_290_tr_100, 70, 70.08))
Moon_290_tr_100.append(find_peaks(Moon_reg_1_lb_290_tr_100, 70, 70.08))
Moon_290_tr_100.append(find_peaks(Moon_reg_0_5_lb_290_tr_100, 70, 70.08))
Moon_290_tr_100.append(find_peaks(Moon_reg_0_1_lb_290_tr_100, 70, 70.08))
Moon_290_tr_100.append(find_peaks(Moon_reg_0_lb_290_tr_100, 70, 70.08))

Moon_45_tr_100.append(find_peaks(Moon_reg_5_lb_45_tr_100, 70, 70.08))
Moon_45_tr_100.append(find_peaks(Moon_reg_1_lb_45_tr_100, 70, 70.08))
Moon_45_tr_100.append(find_peaks(Moon_reg_0_5_lb_45_tr_100, 70, 70.08))
Moon_45_tr_100.append(find_peaks(Moon_reg_0_1_lb_45_tr_100, 70, 70.08))
Moon_45_tr_100.append(find_peaks(Moon_reg_0_lb_45_tr_100, 70, 70.08))

Moon_diff_tr_100 = np.subtract(Moon_290_tr_100, Moon_45_tr_100)

# fig_diff,ax_diff = plt.subplots()

ax_diff.plot(reg_depths, Moon_diff_tr_100, "o-")
ax_diff.set_xlabel("Regolith Depth [m]")
ax_diff.set_ylabel("Temperature Difference [K]")
ax_diff.set_title("Temperature Difference Between 290 K and 45 K Runs")

fig_diff.savefig(
    "Moon_290vs45_diff_tr_100.pdf", 
    format="pdf", 
    bbox_inches=None, 
    dpi=300
)


print(Moon_diff_tr_100)
print(np.mean(Moon_diff_tr_100))
print("stop")


# %%
