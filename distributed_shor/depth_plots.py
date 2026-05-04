import pickle
import matplotlib.pyplot as plt
import numpy as np
from distributed_shor.circ_stat_utils import max_depth, get_avg_CU_durations
from distributed_shor.depth_experiments import get_start_stop_duration_heron
from distributed_shor.weight_utils import (
    get_mono_upper_bound,
    get_start_stop_duration,
    get_weights_heron_ebit_duration,
    get_weights_ionq_forte_ebit_duration,
    get_weights_neutral_atom_ebit_duration,
)
import os
from matplotlib.colors import ListedColormap

from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

"""
results:
0: "Setup: Type UCirc"
2: "noise = Bool"
3: "EJPP = 0-3"
4: "type"
5: depths
6: counts
"""

dark = (5 / 255, 23 / 255, 28 / 255)
bright = (254 / 255, 255 / 255, 255 / 255)
moongrey = (214 / 255, 218 / 255, 221 / 255)
guamblue = (29 / 255, 97 / 255, 158 / 255)
osakared = (236 / 255, 97 / 255, 159 / 255)
abiskogreen = (106 / 255, 191 / 255, 163 / 255)
erfoudorange = (247 / 255, 167 / 255, 18 / 255)

color_lst = [guamblue, osakared, abiskogreen, erfoudorange, guamblue]
color_lst2 = ["grey", "darkgrey", "black"]
ls = ["dashed", "dotted", "dashed"]


color_lst_dlog = [
    "#648FFF",  # Blue
    "#785EF0",  # Purple
    "#DC267F",  # Magenta
    "#FE6100",  # Orange
    "#FFB000",  # Yellow
]

color_lst_std = [
    "#648FFF",  # Blue
    # "#785EF0",  # Purple
    "#DC267F",  # Magenta
    "#FFB000",  # Yellow
]

MARKERS = "osp*X"


cl_no = color_lst[1:]
cmap = ListedColormap(cl_no)
cmap = LinearSegmentedColormap.from_list("custom_colormap", color_lst_dlog)


cmap2 = LinearSegmentedColormap.from_list("custom_colormap2", color_lst_dlog[1:4])
sampled_colors2 = [cmap2(i / 8) for i in range(9)]  # 0 to 1, 11 steps
color_list_new = [color_lst_dlog[0]] + sampled_colors2 + [color_lst_dlog[4]]
cmap_new = ListedColormap(color_list_new)
bounds = list([i - 0.5 for i in range(len(color_list_new) + 2)])
# # Customize global font sizes for specific elements
# plt.rcParams.update(
#     {
#         "axes.titlesize": 20,  # Title font size
#         "axes.labelsize": 18,  # Axis labels font size
#         "xtick.labelsize": 15,  # X-axis tick labels font size
#         "ytick.labelsize": 15,  # Y-axis tick labels font size
#         "legend.fontsize": 13,  # Legend font size
#         "figure.titlesize": 20,  # Figure title font size
#         "legend.title_fontsize": 15,
#         "font.size": 11,
#         "figure.figsize": (6.4, 3.8),
#     }
# )

FS_FULLPAGE = (7.16, 3.5)
FS_COLUMN = (3.5, 2.5)
FS_FULLPAGE_THREE = (7.16, 2.1)
FS_FULLPAGE_THREE2 = (7.16, 2.2)
plt.rcParams.update(
    {
        "font.family": "Times New Roman",  # Use Times New Roman
        "font.size": 8,  # 8 pt font
        "axes.labelsize": 9,
        # "axes.labelsize": 8,
        # "xtick.labelsize": 8,
        # "ytick.labelsize": 8,
        # "legend.fontsize": 8,
        # "lines.linewidth": 1,
        "figure.dpi": 600,
        "savefig.dpi": 600,
        "figure.figsize": (3.5, 2.5),  # column / half page
        # "figure.figsize": (7.16, 3.5),  # full page
        # "figure.figsize": (2.4, 2),  # third page
        "axes.grid": True,
        "axes.grid.axis": "both",
    }
)
# Latex & Font
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = "\\usepackage{amsmath}\n\\usepackage{amssymb}"
plt.rcParams["axes.formatter.use_mathtext"] = True
# plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = "cmr10"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 8
plt.rcParams["axes.titlesize"] = 10
# Lines and marker
# plt.rcParams["lines.linewidth"] = 0.75
plt.rcParams["lines.markersize"] = 2
plt.rcParams["patch.antialiased"] = True

# Grid
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["grid.linestyle"] = "dashed"

# Legend
plt.rcParams["legend.fontsize"] = 6
plt.rcParams["legend.handlelength"] = 4

# Figure
plt.rcParams["savefig.bbox"] = "tight"


def bar_plot(x_labels, bar_labels, data, ylabel, title):

    x = np.arange(len(x_labels))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout="constrained")

    for i in range(len(bar_labels)):
        counts = data[i]

        offset = width * multiplier
        rects = ax.bar(x + offset, counts, width, label=bar_labels[i])
        ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.

    ax.set_ylabel(ylabel)
    # ax.set_title("")
    ax.set_xticks(x + width, x_labels, rotation=90)
    ax.legend(loc="upper left", ncols=3)
    ax.set_ylim(0, 1.5 * np.max(data))

    # plt.show()
    os.makedirs(title, exist_ok=True)
    fig.savefig(title + ".pdf")


def bar_plot_2(x_labels, bar_labels, data, ylabel, title, path):

    x = np.arange(len(x_labels))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots()

    for i in range(len(bar_labels)):
        counts = data[i]

        offset = width * multiplier
        rects = ax.bar(
            x + offset, counts, width, label=bar_labels[i], color=color_lst[i]
        )
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_title("Circuit Depths for N21B2")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x + offset - 1.5 * width, x_labels, rotation=0)
    legend = ax.legend(
        loc="upper center", ncols=5, fancybox=False  # ,title="EJPP"
    )  # loc="upper left", ncols=3)
    ax.set_ylim(0, 1.5 * np.max(data))
    # ax.set_title(title)
    """
    fig.gca().add_artist(legend)
    fig.text(
        1.02,
        0.5,
        "Legend Title",
        transform=fig.gca().transAxes,
        fontsize="large",
        fontweight="bold",
        va="top",
        ha="center",
    )
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    # plt.show()
    os.makedirs(path, exist_ok=True)
    fig.savefig(path + ".pdf")


def depth_scaling_plot(
    x_vals,
    point_lists,
    ylabel,
    title,
    EJPP_list,
    path,
    xscale="linear",
    approach="alternating",
):
    fig, ax = plt.subplots()
    for i, point_list in reversed(list(enumerate(point_lists))):
        ejpp = EJPP_list[i]
        ax.plot(x_vals, point_list, color=color_lst[ejpp - 1], label=ejpp, marker="o")

    ax.set_ylabel(ylabel)
    legend = ax.legend(fancybox=False, title="EJPP")  # ,loc="upper left", ncols=3)
    ax.set_ylim(0, 1.1 * np.max(point_lists))
    # ax.set_title(title)
    ax.set_xlabel("N")
    ax.set_xscale(xscale, base=2)
    fig.tight_layout()
    # plt.show()
    fig.savefig(path + f"_{xscale}_{approach}.pdf")
    plt.close()


def depth_scaling_plot_relative(
    x_vals,
    point_lists,
    ylabel,
    title,
    EJPP_list,
    path,
    xscale="linear",
    approach="alternating",
):
    fig, ax = plt.subplots()
    for i, point_list in reversed(list(enumerate(point_lists))):
        ejpp = EJPP_list[i]
        if i > 0:
            relative_depth_reduction = (
                1 - (point_list / np.max(point_lists, axis=0))
            ) * 100
            ax.plot(
                x_vals,
                relative_depth_reduction,
                color=color_lst[ejpp - 1],
                label=ejpp,
                marker="o",
            )

    ax.set_ylabel(ylabel)
    # legend = ax.legend(fancybox=False)  # ,title="EJPP",loc="upper left", ncols=3)
    ax.set_ylim(0, 1.1 * 100)
    # ax.set_title(title)
    ax.set_xlabel("N")
    ax.set_xscale(xscale, base=2)
    fig.tight_layout()
    # plt.show()
    fig.savefig(path + f"_{xscale}_{approach}.pdf")
    plt.close()


def depth_scaling_plot_monolithic(
    x_vals, point_lists, ylabel, title, path, label_list, xscale="linear"
):
    fig, ax = plt.subplots()
    for i, point_list in enumerate(point_lists):
        ax.plot(x_vals, point_list, color=color_lst[i], label=label_list[i], marker="o")

    ax.set_ylabel(ylabel)
    legend = (
        ax.legend()
    )  # title="Approach", fancybox=False)  # loc="upper left", ncols=3)
    ax.set_ylim(0, 1.1 * np.max(point_lists))
    # ax.set_title(title)
    ax.set_xlabel("N")
    ax.set_xscale(xscale, base=2)
    fig.tight_layout()
    # plt.show()
    fig.savefig(path + f"_{xscale}.pdf")


def depth_scaling_plot_monolithic_relative(
    x_vals, point_lists, ylabel, title, path, label_list, xscale="linear"
):
    fig, ax = plt.subplots()
    for i, point_list in enumerate(point_lists):
        if i == 0:
            continue
        else:
            relative_depth_reduction = (
                1 - (point_list / np.max(point_lists, axis=0))
            ) * 100
            ax.plot(
                x_vals,
                relative_depth_reduction,
                color=color_lst[i],
                label=label_list[i].replace("_", "-"),
                marker="o",
            )

    ax.set_ylabel(ylabel)
    legend = ax.legend(fancybox=False)  # ,title="Approach",loc="upper left", ncols=3)
    ax.set_ylim(0, 1.1 * 100)
    # ax.set_title(title)
    ax.set_xlabel("N")
    ax.set_xscale(xscale, base=2)
    fig.tight_layout()
    # plt.show()

    fig.savefig(path + f"_{xscale}.pdf")


def run_bar_plots(results, result_path):

    for circ_name, circ_dict in results.items():

        designs = ["regular", "iterative", "alternating"]

        circ_depths = [
            [circ_dict[approach][ejpp]["depth"] for approach in designs]
            for ejpp in range(4)
        ]

        bar_labels = [ejpp for ejpp in range(4)]

        bar_plot_2(
            designs,
            bar_labels,
            circ_depths,
            "Circuit Depth",
            title=f"Depths {circ_name}",
            path=result_path + "plots/" + f"{circ_name}",
        )


def run_scaling_plot(
    circ_names,
    x_vals,
    EJPP_list,
    result_path_short,
    hw_label,
    ebit_time,
    xscale="linear",
    approach="alternating",
):

    results = {}
    result_path = result_path_short + f"depth_{hw_label}_{ebit_time}/"
    with open(result_path + "results.pkl", "rb") as fp:  # Unpickling
        results = pickle.load(fp)

    point_lists = [
        [results[circ_name][approach][ejpp]["depth"] for circ_name in circ_names]
        for ejpp in EJPP_list
    ]
    os.makedirs(result_path + "plots/", exist_ok=True)
    depth_scaling_plot(
        x_vals=x_vals,
        point_lists=point_lists,
        ylabel="Delay (ns)",
        title=f"Scaling {approach} {hw_label} {ebit_time/(10**6)} ms",
        path=result_path + "plots/" + "scaling_delay",
        xscale=xscale,
        approach=approach,
        EJPP_list=EJPP_list,
    )


def run_scaling_plot_relative(
    circ_names,
    x_vals,
    EJPP_list,
    result_path_short,
    hw_label,
    ebit_time,
    xscale="linear",
    approach="alternating",
):
    results = {}
    result_path = result_path_short + f"depth_{hw_label}_{ebit_time}/"
    with open(result_path + "results.pkl", "rb") as fp:  # Unpickling
        results = pickle.load(fp)

    point_lists = [
        [results[circ_name][approach][ejpp]["depth"] for circ_name in circ_names]
        for ejpp in EJPP_list
    ]
    os.makedirs(result_path + "plots/", exist_ok=True)
    depth_scaling_plot_relative(
        x_vals=x_vals,
        point_lists=point_lists,
        ylabel="Relative Delay Reduction (\%)",
        title=f"Scaling {approach} {hw_label} {ebit_time/(10**6)} ms",
        path=result_path + "plots/" + "scaling_delay_relative",
        xscale=xscale,
        approach=approach,
        EJPP_list=EJPP_list,
    )


def run_scaling_plot_monolithic_relative(
    circ_names, x_vals, result_path_short, hw_label, approaches, xscale="linear"
):

    result_path = result_path_short + f"depth_{hw_label}/"

    results = {}

    with open(result_path + "results.pkl", "rb") as fp:  # Unpickling
        results = pickle.load(fp)

    point_lists = [
        [results[circ_name][label][0]["depth"] for circ_name in circ_names]
        for label in approaches
    ]

    os.makedirs(result_path + "plots/", exist_ok=True)

    depth_scaling_plot_monolithic_relative(
        x_vals=x_vals,
        point_lists=point_lists,
        ylabel="Relative Delay Reduction (\%)",
        title="Scaling Depth Reduction Monolithic",
        path=result_path + "plots/" + "scaling_depth_monolithic_relative",
        label_list=approaches,
        xscale=xscale,
    )


def run_scaling_plot_monolithic(
    circ_names, x_vals, result_path_short, hw_label, approaches, xscale="linear"
):
    result_path = result_path_short + f"depth_{hw_label}/"

    results = {}

    with open(result_path + "results.pkl", "rb") as fp:  # Unpickling
        results = pickle.load(fp)

    point_lists = [
        [results[circ_name][label][0]["depth"] for circ_name in circ_names]
        for label in approaches
    ]

    os.makedirs(result_path + "plots/", exist_ok=True)

    depth_scaling_plot_monolithic(
        x_vals=x_vals,
        point_lists=point_lists,
        ylabel="Delay (ns)",
        title="Scaling Depth Monolithic",
        path=result_path + "plots/" + "scaling_depth_monolithic",
        label_list=approaches,
        xscale=xscale,
    )


def depth_scaling_plot_monolithic_combined(
    x_vals, point_lists_dict, ylabel, hw_label_list, path, label_list, xscale="linear"
):

    fig, ax = plt.subplots(1, 3, figsize=FS_FULLPAGE_THREE)

    for j, hw_label in enumerate(hw_label_list):
        point_lists = point_lists_dict[hw_label]
        for i, point_list in enumerate(point_lists):

            ax[j].plot(
                x_vals,
                point_list,
                color=color_lst[i],
                label=label_list[i].replace("_", "-"),
                marker="o",
            )
        if j == 0:
            ax[j].set_ylabel(ylabel)
            legend = ax[
                j
            ].legend()  # title="Approach", fancybox=False)  # loc="upper left", ncols=3)
            ax[j].set_ylim(0, 1.1 * np.max(point_lists))
        # ax.set_title(title)
        ax[j].set_xlabel("N")
        ax[j].set_xscale(xscale, base=2)
        ax[j].text(
            0.5,
            -0.3,
            f"({chr(97 + j)})",
            transform=ax[j].transAxes,
            fontsize=8,
            # color="blue",
            ha="center",
            va="top",
        )
        fig.tight_layout()
    # plt.show()
    fig.savefig(path + f"_{xscale}_combined.pdf")


def run_scaling_plot_monolithic_combined(
    circ_names, x_vals, result_path_short, hw_label_list, approaches, xscale="linear"
):
    point_lists_dict = {hw_label: None for hw_label in hw_label_list}
    for hw_label in hw_label_list:
        result_path = result_path_short + f"depth_{hw_label}/"

        results = {}

        with open(result_path + "results.pkl", "rb") as fp:  # Unpickling
            results = pickle.load(fp)

        point_lists = [
            [results[circ_name][label][0]["depth"] for circ_name in circ_names]
            for label in approaches
        ]
        point_lists_dict[hw_label] = point_lists

    os.makedirs(result_path + "plots/", exist_ok=True)

    depth_scaling_plot_monolithic_combined(
        x_vals=x_vals,
        point_lists_dict=point_lists_dict,
        hw_label_list=hw_label_list,
        ylabel="Delay (ns)",
        path=result_path_short + "plots/" + "scaling_depth_monolithic",
        label_list=approaches,
        xscale=xscale,
    )


def circ_name_to_N(circ_name):
    N = circ_name[1:]
    div_ind = len(circ_name)

    dividers = [str.find(circ_name, "B"), str.find(circ_name, "_")]
    for div in dividers:
        if div >= 0 and div < div_ind:
            div_ind = div

    N = int(circ_name[1:div_ind])
    return N


def ebit_scaling_plot_relative(
    x_vals,
    point_lists,
    ylabel,
    title,
    path,
    ejpp_list,
    thresholds=[60.613, 60.613 * 2],
    use_ms=False,
):
    lines = []
    fig, ax = plt.subplots()

    time_adjust = 10**3
    if use_ms:
        time_adjust = 10**6

    for i, point_list in reversed(list(enumerate(point_lists))):
        ejpp = ejpp_list[i]
        if i > 0:
            relative_depth_reduction = (
                (1 - (point_list / np.max(point_lists, axis=0)))
            ) * 100
            lines.append(
                ax.plot(
                    [x_val / (time_adjust) for x_val in x_vals],
                    relative_depth_reduction,
                    color=color_lst[ejpp - 1],
                    label=ejpp,
                    marker="o",
                )[0]
            )

    if len(thresholds) > 0:
        vlines = []
        for i, t in enumerate(thresholds):
            vlines.append(
                ax.vlines(
                    t,
                    ymin=0,
                    ymax=1.1 * 100,
                    label=f"{i+1}" + r"$\cdot t(CU)$",
                    linestyles=ls[i],
                    color=color_lst2[i],
                )
            )
    ax.set_ylabel(ylabel)
    # legend = plt.legend(
    #     handles=lines,
    #     title="EJPP",
    #     fancybox=False,
    # )  # loc="upper left", ncols=3)
    if len(thresholds) > 0:
        legend2 = plt.legend(handles=vlines, fancybox=False)  # ,loc="lower right")
    # plt.gca().add_artist(legend)
    ax.set_ylim(0, 1.1 * 100)
    # ax.set_title(title)
    if use_ms:
        ax.set_xlabel(r"Ebit Generation Time ($ms$)")
    else:
        ax.set_xlabel(r"Ebit Generation Time ($\mu$s)")
    # ax.set_xscale(xscale, base=2)
    fig.tight_layout()
    # plt.show()
    # os.makedirs(path, exist_ok=True)
    fig.savefig(path + ".pdf")
    plt.close()


def run_ebit_scaling_plot_relative(
    result_path,
    circ_name,
    ebit_times,
    hw_label="depth_heron_ebit",
    ejpp_list=[i for i in range(1, 5)],
    approach="alternating",
    plot_thresholds=False,
    use_ms=False,
):

    point_lists = [[] for ejpp in ejpp_list]
    for ebit_t in ebit_times:
        result_path_ebit = result_path + f"depth_{hw_label}_{ebit_t}/"

        with open(result_path_ebit + "results.pkl", "rb") as fp:  # Unpickling
            results = pickle.load(fp)
            for i, ejpp in enumerate(ejpp_list):
                point_lists[i].append(results[circ_name][approach][ejpp]["depth"])

    os.makedirs(result_path + f"depth_{hw_label}/plots/fixed_N/", exist_ok=True)

    if plot_thresholds:
        result_path_avgcu = result_path + f"depth_{hw_label}/"
        N = circ_name_to_N(circ_name)
        avg_CU_dict = {}
        with open(result_path_avgcu + "avg_CU.pkl", "rb") as fp:  # Unpickling
            avg_CU_dict = pickle.load(fp)

        avg_CU = avg_CU_dict[N]["average_duration"]

        thresholds = [avg_CU / (10**3), 2 * avg_CU / (10**3)]

    else:
        thresholds = []

    ebit_scaling_plot_relative(
        x_vals=ebit_times,
        point_lists=point_lists,
        ylabel="Relative Delay Reduction (\%)",
        title=f"{approach} {circ_name}",
        path=result_path
        + f"depth_{hw_label}/plots/fixed_N/"
        + f"{approach}_{circ_name}_scaling_ebit_relative",
        thresholds=thresholds,
        use_ms=use_ms,
        ejpp_list=ejpp_list,
    )


def ebit_scaling_plot_relative_mitigation(
    x_vals,
    point_lists,
    ylabel,
    path,
    title,
    N,
    ejpp_list,
    vls=True,
    hls=False,
    thresholds=[60.613, 2 * 60.613],
):
    n_CUs = 2 * np.ceil(np.log2(N))
    start_stop_d = get_start_stop_duration_heron(
        reset=True, ebit=False
    )  # heron hardcoding!!
    fig, ax = plt.subplots()

    if hls:
        hlines = []
        thresholds = [1 - 1.0 / n_CUs, 1 - 2.0 / n_CUs]
        for i, t in enumerate(thresholds):
            hlines.append(
                ax.hlines(
                    t,
                    xmin=np.min(x_vals) / (10**3),
                    xmax=np.max(x_vals) / (10**3),
                    label=f"{i+1}" + r"$\cdot d_{ebit}$",
                    linestyles=ls[i],
                    color=color_lst2[i],
                )
            )
        legend = plt.legend(handles=hlines, fancybox=False, loc="lower right")

    if vls:
        vlines = []
        for i, t in enumerate(thresholds):
            vlines.append(
                ax.vlines(
                    t,
                    ymin=0,
                    ymax=1.1 * 100,
                    label=f"{i+1}" + r"$\cdot t(CU)$",
                    linestyles=ls[i],
                    color=color_lst2[i],
                )
            )
        legend = plt.legend(handles=vlines, fancybox=False, loc="lower right")

    lines = []

    for i, point_list in reversed(list(enumerate(point_lists))):
        ejpp = ejpp_list[i]
        if i > 0:
            abs_depth_difference = np.max(point_lists, axis=0) - point_list
            relative_ebit_delay_mitigation = [
                (abs_depth_difference[j] / (n_CUs * (x_vals[j] + start_stop_d))) * 100
                for j in range(len(x_vals))
            ]
            lines.append(
                ax.plot(
                    [x_val / (10**3) for x_val in x_vals],
                    relative_ebit_delay_mitigation,
                    color=color_lst[ejpp - 1],
                    label=ejpp,
                    marker="o",
                )[0]
            )

    ax.set_ylabel(ylabel)
    legend2 = plt.legend(
        handles=lines,
        title="EJPP",
        fancybox=False,
    )  # loc="upper left", ncols=3)

    plt.gca().add_artist(legend)
    ax.set_ylim(0, 1.1 * 100)
    # ax.set_title(title)
    ax.set_xlabel(r"Ebit Generation Time ($\mu$s)")
    # ax.set_xscale(xscale, base=2)
    fig.tight_layout()
    # plt.show()
    # os.makedirs(path, exist_ok=True)
    fig.savefig(path + ".pdf")
    plt.close()


def run_ebit_scaling_plot_relative_mitigation(
    result_path,
    circ_name,
    ebit_times,
    hw_label="depth_heron_ebit",
    ejpp_list=[i for i in range(1, 5)],
    approach="alternating",
):

    point_lists = [[] for ejpp in ejpp_list]
    for ebit_t in ebit_times:
        result_path_ebit = result_path + f"depth_{hw_label}_{ebit_t}/"

        with open(result_path_ebit + "results.pkl", "rb") as fp:  # Unpickling
            results = pickle.load(fp)
            for i, ejpp in enumerate(ejpp_list):
                point_lists[i].append(results[circ_name][approach][ejpp]["depth"])

    result_path_avgcu = result_path + f"depth_{hw_label}/"
    N = circ_name_to_N(circ_name)
    avg_CU_dict = {}
    with open(result_path_avgcu + "avg_CU.pkl", "rb") as fp:  # Unpickling
        avg_CU_dict = pickle.load(fp)

    avg_CU = avg_CU_dict[N]["average_duration"]

    thresholds = [avg_CU / (10**3), 2 * avg_CU / (10**3)]

    os.makedirs(result_path + f"depth_{hw_label}/plots/fixed_N/", exist_ok=True)

    ebit_scaling_plot_relative_mitigation(
        x_vals=ebit_times,
        point_lists=point_lists,
        ylabel="Relative Ebit Time \n Mitigation (\%)",
        title=f"{approach} {circ_name}",
        N=N,
        path=result_path
        + f"depth_{hw_label}/plots/fixed_N/"
        + f"{approach}_{circ_name}_scaling_ebit_relative_mitigation",
        thresholds=thresholds,
        ejpp_list=ejpp_list,
    )


def heatmap_plot(x_vals, y_vals, point_array, ylabel, title, path):
    fig, ax = plt.subplots()
    # im = ax.imshow(point_array.transpose(), origin="lower")
    # bounds = list([i - 0.5 for i in range(13)])
    im = ax.pcolormesh(
        x_vals,
        y_vals,
        point_array.transpose(),
        shading="nearest",
        vmin=1,
        vmax=11,  # np.max(point_array),
        cmap=cmap,
    )
    cbar = ax.figure.colorbar(
        im,
        ax=ax,
        boundaries=bounds[1:],
        ticks=[i for i in range(1, 5)] + [6, 8, 10] + [11],
    )  # , ticks=np.arange(1, 5))
    cbar.ax.set_ylabel("Optimal EJPP", rotation=-90, va="bottom")
    cbar.ax.set_yticklabels([i for i in range(1, 5)] + [6, 8, 10] + [r"$>$ 10"])

    # Show all ticks and label them with the respective list entries
    # ax.set_xticks(range(len(x_vals)), labels=np.ceil(np.log2(x_vals)))
    # ax.set_yticks(range(len(y_vals)), labels=y_vals)

    ax.set_ylabel(ylabel)

    ax.set_xlabel("N")

    # ax.set_title(title)
    ax.set_xscale("log", base=2)
    fig.tight_layout()
    fig.savefig(path + ".pdf")
    plt.close()


def heatmap_plot_combined_hw(
    x_vals, y_vals_dict, point_array_dict, hw_list, ylabel, title, path
):
    # fig, axes = plt.subplots(1, 3, figsize=FS_FULLPAGE_THREE2)
    fig = plt.figure(figsize=FS_FULLPAGE_THREE2, constrained_layout=True)
    gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.05])

    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    for j, ax in enumerate(axes):
        # im = ax.imshow(point_array.transpose(), origin="lower")
        im = ax.pcolormesh(
            x_vals,
            y_vals_dict[hw_list[j]],
            point_array_dict[hw_list[j]].transpose(),
            shading="nearest",
            vmin=1,
            vmax=11,
            cmap=cmap,
        )

        # Show all ticks and label them with the respective list entries
        # ax.set_xticks(range(len(x_vals)), labels=np.ceil(np.log2(x_vals)))
        # ax.set_yticks(range(len(y_vals)), labels=y_vals)
        if j == 0:
            ax.set_ylabel(ylabel)

        ax.set_xscale("log", base=2)
        ax.text(
            0.5,
            -0.4,
            f"({chr(97 + j)})",
            transform=ax.transAxes,
            fontsize=8,
            # color="blue",
            ha="center",
            va="top",
        )
        ax.set_xlabel("N")

        # ax.set_title(title)
    # divider = make_axes_locatable(axes[2])
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(
        im,
        cax=cax,
        boundaries=bounds[1:],
        ticks=[i for i in range(1, 5)] + [6, 8, 10] + [11],
    )  # , ticks=np.arange(1, 5))
    cbar.ax.set_ylabel("Optimal EJPP", rotation=-90, va="bottom")
    cbar.ax.set_yticklabels([i for i in range(1, 5)] + [6, 8, 10] + [r"$>$ 10"])
    # fig.tight_layout()
    plt.savefig(path + ".pdf")
    plt.close()


def run_heatmap_plot_combined_hw(
    result_path_root,
    x_vals,
    ebit_times_dict,
    hw_list,
    EJPP_list=range(2, 5),
    approach="alternating",
    margin_scale=None,
):
    ejpp_opt_dict = {hw_label: None for hw_label in hw_list}
    for hw_label in hw_list:
        ebit_times = ebit_times_dict[hw_label]
        ejpp_opt_im = np.zeros((len(x_vals), len(ebit_times)))

        for j, ebit_t in enumerate(ebit_times):
            result_path_ebit = result_path_root + f"depth_{hw_label}_{ebit_t}/"

            with open(result_path_ebit + "results.pkl", "rb") as fp:  # Unpickling
                results = pickle.load(fp)
                for i, N in enumerate(x_vals):
                    circ_name = f"N{N}_QRISP"
                    ejpp_opt = 1
                    d_opt = results[circ_name][approach][1]["depth"]
                    if margin_scale:
                        margin = d_opt / (
                            500 / (1 / 2.5 * np.min([np.ceil(np.log2(N)), 20]))
                        )
                    else:
                        margin = d_opt / (
                            6 * np.ceil(np.log2(N))
                        )  # 10% of average CU+dist+phase duration

                    for ejpp in EJPP_list:
                        d_ejpp = results[circ_name][approach][ejpp]["depth"]
                        if d_ejpp + margin < d_opt:
                            d_opt = d_ejpp
                            ejpp_opt = ejpp

                    ejpp_opt_im[i][j] = ejpp_opt

        ejpp_opt_dict[hw_label] = ejpp_opt_im

    heatmap_plot_combined_hw(
        x_vals=x_vals,
        y_vals_dict=ebit_times_dict,
        point_array_dict=ejpp_opt_dict,
        hw_list=hw_list,
        ylabel=r"Ebit Generation Time ($\mu$s)",
        title=f"Optimal EJPP",
        path=result_path_root + f"/{approach}_heatmap_combined_approach",
    )


def heatmap_plot_combined_approach(
    x_vals, y_vals, point_array_dict, approach_list, ylabel, title, path
):

    fig = plt.figure(figsize=FS_FULLPAGE_THREE2, constrained_layout=True)
    gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 0.05])

    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    for j, ax in enumerate(axes):
        # im = ax.imshow(point_array.transpose(), origin="lower")
        im = ax.pcolormesh(
            x_vals,
            y_vals,
            point_array_dict[approach_list[j]].transpose(),
            shading="nearest",
            vmin=1,
            vmax=11,
            cmap=cmap,
        )
        if j == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.tick_params(labelleft=False)

        ax.set_xlabel("N")
        ax.text(
            0.5,
            -0.4,
            f"({chr(97 + j)})",
            transform=ax.transAxes,
            fontsize=8,
            # color="blue",
            ha="center",
            va="top",
        )
        # ax.set_title(title)
        ax.set_xscale("log", base=2)

    cbar = fig.colorbar(
        im,
        cax=cax,
        boundaries=bounds[1:],
        ticks=[i for i in range(1, 5)] + [6, 8, 10] + [11],
    )  # , ticks=np.arange(1, 5))
    cbar.ax.set_ylabel("Optimal EJPP", rotation=-90, va="bottom")
    cbar.ax.set_yticklabels([i for i in range(1, 5)] + [6, 8, 10] + [r"$>$ 10"])

    fig.savefig(path + ".pdf")
    plt.close()


def run_heatmap_plot_combined_approach(
    result_path_root,
    x_vals,
    ebit_times,
    EJPP_list=range(2, 5),
    hw_label="depth_heron_ebit",
    approach_list=["alternating"],
    margin_scale=False,
):
    ejpp_opt_dict = {approach: None for approach in approach_list}
    for approach in approach_list:
        ejpp_opt_im = np.zeros((len(x_vals), len(ebit_times)))

        for j, ebit_t in enumerate(ebit_times):
            result_path_ebit = result_path_root + f"depth_{hw_label}_{ebit_t}/"

            with open(result_path_ebit + "results.pkl", "rb") as fp:  # Unpickling
                results = pickle.load(fp)
                for i, N in enumerate(x_vals):
                    circ_name = f"N{N}_QRISP"
                    ejpp_opt = 1
                    d_opt = results[circ_name][approach][1]["depth"]

                    if margin_scale:
                        margin = d_opt / (
                            500 / (1 / 2.5 * np.min([np.ceil(np.log2(N)), 20]))
                        )
                    else:
                        margin = d_opt / (
                            6 * np.ceil(np.log2(N))
                        )  # 10% of average CU+dist+phase duration

                    for ejpp in EJPP_list:
                        d_ejpp = results[circ_name][approach][ejpp]["depth"]
                        if d_ejpp + margin < d_opt:
                            d_opt = d_ejpp
                            ejpp_opt = ejpp

                    ejpp_opt_im[i][j] = ejpp_opt

        ejpp_opt_dict[approach] = ejpp_opt_im

    heatmap_plot_combined_approach(
        x_vals=x_vals,
        y_vals=ebit_times,
        point_array_dict=ejpp_opt_dict,
        approach_list=approach_list,
        ylabel=r"Ebit Generation Time ($\mu$s)",
        title=f"Optimal EJPP {hw_label}",
        path=result_path_root
        + f"depth_{hw_label}"
        + f"/{hw_label}_heatmap_combined_approach",
    )


def run_heatmap_plot_bound(
    result_path_root,
    x_vals,
    ebit_times,
    weight_func,
    hw_label="depth_heron_ebit",
    dlog=False,
):

    ejpp_opt_im = np.zeros((len(x_vals), len(ebit_times)))

    avg_CU_durations = get_avg_CU_durations(
        result_path=result_path_root, weight_label=hw_label, x_vals=x_vals
    )

    for j, ebit_t in enumerate(ebit_times):
        dqc_d = get_start_stop_duration(weight_func(ebit_t))
        for i, N in enumerate(x_vals):
            CU_d = avg_CU_durations[i]
            CU_count = 2 * np.ceil(np.log2(N))
            if dlog:
                CU_count = np.ceil(1.5 * np.ceil(np.log2(N)))
            ejpp_opt_im[i][j] = min(np.ceil((dqc_d / CU_d) + 1), CU_count, 11)

    heatmap_plot(
        x_vals=x_vals,
        y_vals=ebit_times,
        point_array=ejpp_opt_im,
        ylabel=r"Ebit Generation Time ($\mu$s)",
        title=f"Optimal EJPP {hw_label}",
        path=result_path_root + f"depth_{hw_label}" + f"/{hw_label}_heatmap_bound",
    )


def run_heatmap_plot(
    result_path_root,
    x_vals,
    ebit_times,
    EJPP_list=range(2, 5),
    hw_label="depth_heron_ebit",
    approach="alternating",
    margin_scale=False,
):

    ejpp_opt_im = np.zeros((len(x_vals), len(ebit_times)))

    for j, ebit_t in enumerate(ebit_times):
        result_path_ebit = result_path_root + f"depth_{hw_label}_{ebit_t}/"

        with open(result_path_ebit + "results.pkl", "rb") as fp:  # Unpickling
            results = pickle.load(fp)
            for i, N in enumerate(x_vals):
                circ_name = f"N{N}_QRISP"
                ejpp_opt = 1
                d_opt = results[circ_name][approach][1]["depth"]
                # margin_scale = False
                if margin_scale:
                    margin = d_opt * 0.03
                else:
                    margin = d_opt / (
                        6 * np.ceil(np.log2(N))
                    )  # 10% of average CU+dist+phase duration
                # margin = d_opt / 10
                for ejpp in EJPP_list:
                    d_ejpp = results[circ_name][approach][ejpp]["depth"]
                    if d_ejpp + margin < d_opt:
                        d_opt = d_ejpp
                        ejpp_opt = ejpp

                ejpp_opt_im[i][j] = ejpp_opt

    heatmap_plot(
        x_vals=x_vals,
        y_vals=ebit_times,
        point_array=ejpp_opt_im,
        ylabel=r"Ebit Generation Time ($\mu$s)",
        title=f"Optimal EJPP {hw_label}",
        path=result_path_root + f"depth_{hw_label}" + f"/{hw_label}_{approach}_heatmap",
    )


def CU_scaling_plot(x_vals, point_list, path, ylabel, title, xscale="log"):

    fig, ax = plt.subplots()
    ax.plot(x_vals, point_list, color=color_lst[-1], marker="o")

    ax.set_ylabel(ylabel)
    # legend = ax.legend(fancybox=False)  # loc="upper left", ncols=3)
    ax.set_ylim(0, 1.1 * np.max(point_list))
    # ax.set_title(title)
    ax.set_xlabel("N")
    ax.set_xscale(xscale, base=2)
    fig.tight_layout()
    # plt.show()
    fig.savefig(path + f"_{xscale}.pdf")


def CU_scaling_plot_list(
    x_vals, point_lists, weight_list, path, ylabel, title, xscale="log"
):

    fig, ax = plt.subplots()
    for i in range(len(point_lists)):
        ax.plot(
            x_vals,
            point_lists[i],
            color=color_lst[i],
            label=weight_list[i].label,
            marker="o",
        )

    ax.set_ylabel(ylabel)
    legend = ax.legend(fancybox=False)  # loc="upper left", ncols=3)
    # ax.set_ylim(0, 1.1 * np.max([np.max(point_list) for point_list in point_lists]))
    # ax.set_title(title)
    ax.set_xlabel("N")
    ax.set_yscale("log")
    ax.set_xscale(xscale, base=2)
    fig.tight_layout()
    # plt.show()
    fig.savefig(path + f"_{xscale}.pdf")


def run_CU_scaling_plot(result_path):

    results = {}

    weight_label = ""
    with open(result_path + "weights.pkl", "rb") as fp:  # Unpickling
        weights = pickle.load(fp)
        weight_label = weights.label
    with open(result_path + "avg_CU.pkl", "rb") as fp:  # Unpickling
        results = pickle.load(fp)

    x_vals = [n for n in results.keys()]
    avg_CU_durations = [results[x]["average_duration"] for x in x_vals]
    work_reg_sizes = [results[x]["work_reg_size"] for x in x_vals]

    os.makedirs(result_path + "plots/", exist_ok=True)
    CU_scaling_plot(
        x_vals=x_vals,
        point_list=avg_CU_durations,
        path=result_path + "plots/" + f"{weight_label}_average_CU_duration",
        ylabel="average CU duration",
        title=f"Average CU duration for {weight_label}",
    )
    CU_scaling_plot(
        x_vals=x_vals,
        point_list=work_reg_sizes,
        path=result_path + "plots/" + f"{weight_label}_work_reg_sizes",
        ylabel="Work Register Size",
        title=f"Work Register Size for {weight_label}",
    )


def run_CU_scaling_plot_list(result_path, weight_label_list, x_vals):

    avg_CU_durations_list = []
    weight_list = []
    for weight_label in weight_label_list:
        results = {}

        with open(
            result_path + f"depth_{weight_label}/" + "weights.pkl", "rb"
        ) as fp:  # Unpickling
            weights = pickle.load(fp)
        with open(
            result_path + f"depth_{weight_label}/" + "avg_CU.pkl", "rb"
        ) as fp:  # Unpickling
            results = pickle.load(fp)

        avg_CU_durations_list.append([results[x]["average_duration"] for x in x_vals])
        weight_list.append(weights)

    os.makedirs(result_path + "plots/", exist_ok=True)
    CU_scaling_plot_list(
        x_vals=x_vals,
        point_lists=avg_CU_durations_list,
        weight_list=weight_list,
        path=result_path + "plots/" + f"average_CU_duration_list",
        ylabel="average CU duration",
        title=f"Average CU duration",
    )


def inst_count_plot(inst_count_list, x_vals, path):
    fig, ax = plt.subplots()

    ax.plot(  # q1 counts
        x_vals,
        inst_count_list[0],
        color=color_lst[0],
        marker="o",
        label=r" $\# U_{1q}$",
    )
    ax.plot(  # q2 counts
        x_vals,
        inst_count_list[1],
        color=color_lst[1],
        ls="--",
        marker="o",
        label=r" $\# U_{2q}$",
    )

    ax.set_ylabel("Instruction-count")
    legend = ax.legend(fancybox=False)  # loc="upper left", ncols=3)
    # ax.set_ylim(0, 1.1 * np.max(point_list))
    # ax.set_title("Monolithic Instruction-counts")
    ax.set_xlabel("N")
    ax.set_xscale("log", base=2)
    # ax.set_yscale("log", base=10)
    fig.tight_layout()
    # plt.show()
    fig.savefig(path + f"inst_count.pdf")


def run_inst_count_plot(weight_label, x_vals, path):

    inst_count_list = []

    with open(path + f"depth_{weight_label}/" + "avg_CU.pkl", "rb") as fp:  # Unpickling
        avg_CU_durations = pickle.load(fp)

        q_1_list = [avg_CU_durations[N]["inst_per_length"][1] for N in x_vals]
        q_2_list = [avg_CU_durations[N]["inst_per_length"][2] for N in x_vals]
        inst_count_list = [q_1_list, q_2_list]

    inst_count_plot(
        inst_count_list=inst_count_list,
        x_vals=x_vals,
        path=path,
    )


def monolithic_scaling_plot_relative_mitigation(
    x_vals,
    point_lists,
    ylabel,
    path,
    weights,
    label_list,
    title="Monolithic Mitigation",
    bound_scale=1.0,
    vls=False,
    hls=False,
):
    n_CUs = 2 * np.ceil(np.log2(x_vals))

    fig, ax = plt.subplots()

    if hls:
        hlines = []
        thresholds = [1 - 1.0 / n_CUs, 1 - 2.0 / n_CUs]
        for i, t in enumerate(thresholds):
            hlines.append(
                ax.hlines(
                    t,
                    xmin=np.min(x_vals) / (10**3),
                    xmax=np.max(x_vals) / (10**3),
                    label=f"{i+1}" + r"$\cdot d_{ebit}$",
                    linestyles=ls[i],
                    color=color_lst2[i],
                )
            )
        legend = plt.legend(handles=hlines, fancybox=False, loc="lower right")

    if vls:
        thresholds = [60.613, 2 * 60.613, 3 * 60.613]
        vlines = []
        for i, t in enumerate(thresholds):
            vlines.append(
                ax.vlines(
                    t,
                    ymin=0,
                    ymax=1.1,
                    label=f"{i+1}" + r"$\cdot t(CU)$",
                    linestyles=ls[i],
                    color=color_lst2[i],
                )
            )
        legend = plt.legend(handles=vlines, fancybox=False, loc="lower right")

    lines = []

    for i, point_list in enumerate(point_lists):
        if i == 0:
            continue
        else:
            abs_depth_difference = np.max(point_lists, axis=0) - point_list
            relative_mono_delay_mitigation = [
                (
                    abs_depth_difference[j]
                    / (bound_scale * get_mono_upper_bound(weights, n_CUs[j]))
                )
                * 100
                for j in range(len(x_vals))
            ]
            lines.append(
                ax.plot(
                    x_vals,
                    relative_mono_delay_mitigation,
                    color=color_lst[i],
                    label=label_list[i].replace("_", "-"),
                    marker="o",
                )[0]
            )

    ax.set_ylabel(ylabel)
    legend2 = plt.legend(
        handles=lines,
        # title="EJPP",
        fancybox=False,
    )  # loc="upper left", ncols=3)

    # plt.gca().add_artist(legend)
    ax.set_ylim(0, 1.1 * 100)
    # ax.set_title(title)
    ax.set_xlabel("N")
    ax.set_xscale("log", base=2)
    fig.tight_layout()
    # plt.show()
    fig.savefig(path + ".pdf")


def run_monolithic_scaling_plot_relative_mitigation(
    result_path, circ_names, x_vals, approach_list, hw_label="depth_heron_ebit"
):

    point_lists = [[] for _ in approach_list]

    result_path_mono = result_path + f"depth_{hw_label}/"

    with open(result_path_mono + "results.pkl", "rb") as fp:  # Unpickling
        results = pickle.load(fp)
        point_lists = [
            [results[circ_name][approach][0]["depth"] for circ_name in circ_names]
            for approach in approach_list
        ]

    weights = None
    with open(result_path_mono + "weights.pkl", "rb") as fp:  # Unpickling
        weights = pickle.load(fp)

    path_plot_folder = result_path + "plots/"
    os.makedirs(path_plot_folder, exist_ok=True)

    monolithic_scaling_plot_relative_mitigation(
        x_vals=x_vals,
        point_lists=point_lists,
        path=path_plot_folder + f"{hw_label}_scaling_mono_relative_mitigation",
        ylabel="Relative Monolithic \n Bound Mitigation (\%)",
        title=f"Mitigation {hw_label}",
        label_list=approach_list,
        weights=weights,
    )


def ebit_scaling_plot(
    x_vals,
    point_lists,
    ylabel,
    title,
    path,
    ejpp_list,
    thresholds=[60.613, 60.613 * 2],
    vlines=False,
):
    lines = []
    fig, ax = plt.subplots()
    for i, point_list in reversed(list(enumerate(point_lists))):
        ejpp = ejpp_list[i]
        lines.append(
            ax.plot(
                [x_val / (10**3) for x_val in x_vals],
                point_list,
                color=color_lst[ejpp - 1],
                label=ejpp,
                marker="o",
            )[0]
        )

    if vlines:
        vlines = []
        for i, t in enumerate(thresholds):
            vlines.append(
                ax.vlines(
                    t,
                    ymin=0,
                    ymax=1,
                    label=f"{i+1}" + r"$\cdot t(CU)$",
                    linestyles=ls[i],
                    color=color_lst2[i],
                )
            )

    legend = plt.legend(
        handles=lines,
        title="EJPP",
        fancybox=False,
    )  # loc="upper left", ncols=3)
    if vlines:
        legend2 = plt.legend(handles=vlines, fancybox=False)
    ax.set_ylabel(ylabel)

    plt.gca().add_artist(legend)
    # ax.set_title(title)
    ax.set_xlabel(r"Ebit Generation Time ($\mu$s)")
    # ax.set_xscale(xscale, base=2)
    fig.tight_layout()
    # plt.show()
    fig.savefig(path + ".pdf")
    plt.close()


def run_ebit_scaling_plot(
    result_path,
    circ_name,
    ebit_times,
    hw_label="depth_heron_ebit",
    ejpp_list=[i for i in range(1, 5)],
    approach="alternating",
):

    point_lists = [[] for ejpp in ejpp_list]
    for ebit_t in ebit_times:
        result_path_ebit = result_path + f"depth_{hw_label}_{ebit_t}/"

        with open(result_path_ebit + "results.pkl", "rb") as fp:  # Unpickling
            results = pickle.load(fp)
            for i, ejpp in enumerate(ejpp_list):
                point_lists[i].append(results[circ_name][approach][ejpp]["depth"])

    os.makedirs(result_path + f"depth_{hw_label}/plots/fixed_N/", exist_ok=True)

    ebit_scaling_plot(
        x_vals=ebit_times,
        point_lists=point_lists,
        ylabel="Delay (ns)",
        title=f"{approach} {circ_name}",
        path=result_path
        + f"depth_{hw_label}/plots/fixed_N/"
        + f"{approach}_{hw_label}_{circ_name}_scaling_ebit",
        ejpp_list=ejpp_list,
    )


def qrisp_scaling_plot_combined(
    x_vals,
    work_reg_sizes,
    avg_CU_durations_list,
    weight_list,
    inst_count_list,
    ylabels,
    path,
    xscale="log",
):

    fig, axes = plt.subplots(1, 3, figsize=FS_FULLPAGE_THREE)

    for j, ax in enumerate(axes):
        if j == 0:
            ax.plot(x_vals, work_reg_sizes, color=color_lst[0], marker="o")
            ax.set_ylim(0, 1.1 * np.max(work_reg_sizes))
        if j == 1:
            ax.plot(  # q1 counts
                x_vals,
                inst_count_list[0],
                color=color_lst[0],
                marker="o",
                label=r" $\# U_{1q}$",
            )
            ax.plot(  # q2 counts
                x_vals,
                inst_count_list[1],
                color=color_lst[1],
                # ls="--",
                marker="o",
                label=r" $\# U_{2q}$",
            )
            legend = ax.legend(fancybox=False)
        if j == 2:
            for i in range(len(avg_CU_durations_list)):
                ax.plot(
                    x_vals,
                    avg_CU_durations_list[i],
                    color=color_lst[i],
                    label=weight_list[i].label,
                    marker="o",
                )
            ax.set_yscale("log")
            ax.set_ylim(10**2, 10**11)
            legend = ax.legend(fancybox=False)
        ax.set_ylabel(ylabels[j])
        # ax.set_title(title)
        ax.set_xlabel("N")
        ax.set_xscale(xscale, base=2)
        ax.text(
            0.5,
            -0.3,
            f"({chr(97 + j)})",
            transform=ax.transAxes,
            fontsize=8,
            # color="blue",
            ha="center",
            va="top",
        )
        fig.tight_layout()
    # plt.show()
    fig.savefig(path + f"_{xscale}_combined.pdf")


def qrisp_circ_stats_plots_combined(x_vals):

    result_path = (
        "./src/semi_iterative_comparison/results_circ_stats/depth_neutral_atom/"
    )

    result_path_short = "./src/semi_iterative_comparison/results_circ_stats/"
    weight_label_list = ["neutral_atom", "ionq_forte", "heron"]

    results = {}

    with open(result_path + "avg_CU.pkl", "rb") as fp:  # Unpickling
        results = pickle.load(fp)

    x_vals = [n for n in results.keys()]
    work_reg_sizes = [results[x]["work_reg_size"] for x in x_vals]

    avg_CU_durations_list = []
    weight_list = []
    for weight_label in weight_label_list:
        results = {}

        with open(
            result_path_short + f"depth_{weight_label}/" + "weights.pkl", "rb"
        ) as fp:  # Unpickling
            weights = pickle.load(fp)
        with open(
            result_path_short + f"depth_{weight_label}/" + "avg_CU.pkl", "rb"
        ) as fp:  # Unpickling
            results = pickle.load(fp)

        avg_CU_durations_list.append([results[x]["average_duration"] for x in x_vals])
        weight_list.append(weights)

    os.makedirs(result_path_short + "plots/", exist_ok=True)

    inst_count_list = []

    with open(
        result_path_short + f"depth_{weight_label_list[0]}/" + "avg_CU.pkl", "rb"
    ) as fp:  # Unpickling
        avg_CU_durations = pickle.load(fp)

        q_1_list = [avg_CU_durations[N]["inst_per_length"][1] for N in x_vals]
        q_2_list = [avg_CU_durations[N]["inst_per_length"][2] for N in x_vals]
        inst_count_list = [q_1_list, q_2_list]

    qrisp_scaling_plot_combined(
        x_vals=x_vals,
        work_reg_sizes=work_reg_sizes,
        path=result_path_short + "plots/" + f"qrisp_stats_combined",
        ylabels=[
            "Work Register Size (qubits)",
            "Instruction-Count",
            r"Average $t(CU)$ (ns)",
        ],
        avg_CU_durations_list=avg_CU_durations_list,
        weight_list=weight_list,
        inst_count_list=inst_count_list,
    )


def qrisp_circ_stats_plots(x_vals):
    result_path = (
        "./src/semi_iterative_comparison/results_circ_stats/depth_neutral_atom/"
    )
    run_CU_scaling_plot(result_path)

    result_path_short = "./src/semi_iterative_comparison/results_circ_stats/"
    weight_label_list = ["neutral_atom", "ionq_forte", "heron"]
    run_CU_scaling_plot_list(
        result_path=result_path_short,
        weight_label_list=weight_label_list,
        x_vals=x_vals,
    )

    run_inst_count_plot(
        path=result_path_short,
        weight_label=weight_label_list[0],
        x_vals=x_vals,
    )


def dist_fixed_ebit_plots(
    hw_labels,
    ebit_times_dict,
    approaches,
    result_path_short,
    circ_names,
    x_vals,
    EJPP_list,
):
    print("Distributed plots, fixed ebit_times:")
    for hw_label in hw_labels:
        ebit_times = ebit_times_dict[hw_label]
        for ebit_time in ebit_times:
            print(f"Plotting {hw_label}, {ebit_time}")
            for approach in approaches:

                run_scaling_plot(
                    result_path_short=result_path_short,
                    hw_label=hw_label,
                    ebit_time=ebit_time,
                    xscale="log",
                    circ_names=circ_names,
                    x_vals=x_vals,
                    EJPP_list=EJPP_list,
                    approach=approach,
                )
                run_scaling_plot_relative(
                    result_path_short=result_path_short,
                    hw_label=hw_label,
                    ebit_time=ebit_time,
                    xscale="log",
                    circ_names=circ_names,
                    x_vals=x_vals,
                    EJPP_list=EJPP_list,
                    approach=approach,
                )


def mono_plots(hw_labels_mono, result_path_short, circ_names, approaches, x_vals):

    print("Monolithic plots:")

    print("Combined monolithic plots")
    run_scaling_plot_monolithic_combined(
        result_path_short=result_path_short,
        xscale="log",
        circ_names=circ_names,
        approaches=approaches,
        x_vals=x_vals,
        hw_label_list=hw_labels_mono,
    )

    for hw_label in hw_labels_mono:
        print(f"Scaling runtime monolithic {hw_label}")
        run_scaling_plot_monolithic(
            result_path_short=result_path_short,
            hw_label=hw_label,
            xscale="log",
            circ_names=circ_names,
            approaches=approaches,
            x_vals=x_vals,
        )

        print(f"Scaling runtime relative monolithic {hw_label}")
        run_scaling_plot_monolithic_relative(
            result_path_short=result_path_short,
            hw_label=hw_label,
            xscale="log",
            circ_names=circ_names,
            approaches=approaches,
            x_vals=x_vals,
        )

        print(f"Scaling runtime relative mitigation {hw_label}")
        run_monolithic_scaling_plot_relative_mitigation(
            result_path=result_path_short,
            circ_names=circ_names,
            x_vals=x_vals,
            approach_list=approaches,
            hw_label=hw_label,
        )


def dist_fixed_N_plots(
    hw_labels,
    ebit_times_dict,
    circ_names,
    approaches,
    EJPP_list,
    result_path_short,
    use_ms=False,
    plot_thresholds=False,
):

    print("Distributed plots, fixed N:")

    for hw_label in hw_labels:
        ebit_times = ebit_times_dict[hw_label]
        for circ_name in circ_names:

            print(f"Plotting {hw_label}, {circ_name}")
            for approach in approaches:

                run_ebit_scaling_plot(
                    result_path=result_path_short,
                    ebit_times=ebit_times,
                    circ_name=circ_name,
                    hw_label=hw_label,
                    ejpp_list=EJPP_list,
                    approach=approach,
                )

                run_ebit_scaling_plot_relative(
                    result_path=result_path_short,
                    ebit_times=ebit_times,
                    circ_name=circ_name,
                    hw_label=hw_label,
                    approach=approach,
                    ejpp_list=EJPP_list,
                    plot_thresholds=plot_thresholds,
                    use_ms=use_ms,
                )

                run_ebit_scaling_plot_relative_mitigation(
                    result_path=result_path_short,
                    ebit_times=ebit_times,
                    circ_name=circ_name,
                    hw_label=hw_label,
                    ejpp_list=EJPP_list,
                )


def default_shor_plots():

    circ_names = [
        "N15_QRISP",
        "N21_QRISP",
        "N25_QRISP",
        # "N33_QRISP",
        # "N35_QRISP",
        # "N39_QRISP",
        # "N45_QRISP",
        # "N49_QRISP",
        # "N71_QRISP",
        # "N711_QRISP",
        # "N1311_QRISP",
        # "N3111_QRISP",
        # "N7111_QRISP",
        # "N13111_QRISP",
        # "N31111_QRISP",
        # "N41111_QRISP",
        # "N71111_QRISP",
        # "N141111_QRISP",
        # "N411111_QRISP",
        # "N711111_QRISP",
        # "N1311111_QRISP",
        # "N13111111_QRISP",
        # "N141111111_QRISP",
        # "N1411111111_QRISP",
        # "N13111111111_QRISP",
        # "N131111111111_QRISP",
        # "N711111111111_QRISP",
        # "N33111111111111_QRISP",
        # "N711111111111111_QRISP",
        # "N711111111111111111_QRISP",
        # "N13111111111111111111_QRISP",
    ]

    x_vals = [
        15,
        21,
        25,
        # 33,
        # 35,
        # 39,
        # 45,
        # 49,
        # 71,
        # 711,
        # 1311,
        # 3111,
        # 7111,
        # 13111,
        # 31111,
        # 41111,
        # 71111,
        # 141111,
        # 411111,
        # 711111,
        # 1311111,
        # 13111111,
        # 141111111,
        # 1411111111,
        # 13111111111,
        # 131111111111,
        # 711111111111,
        # 33111111111111,
        # 711111111111111,
        # 711111111111111111,
        # 13111111111111111111,
    ]

    EJPP_list = [1, 2, 4]  # [i for i in range(1, 5)] + [6, 8, 10]

    ebit_list_ion = [
        (ebit_d_ms * 10**6) for ebit_d_ms in range(5, 20000, 500)
    ]  # 5ms to 20s

    ebit_list_sc = [(ebit_d_mus * 10**3) for ebit_d_mus in range(10, 1000, 20)]
    ebit_list_natom = [(ebit_d_ms * 10**6) for ebit_d_ms in range(5, 120, 20)]
    # [(ebit_d_ms * 10**6) for ebit_d_ms in range(5, 120, 5)]

    result_path_short = "./out/results/"
    ebit_times_dict = {
        "neutral_atom": ebit_list_natom,
        "heron_ebit": ebit_list_sc,
        "ionq_forte": ebit_list_ion,
        "heron": ebit_list_sc,
    }

    weight_func_dict = {
        "neutral_atom": get_weights_neutral_atom_ebit_duration,
        "heron": get_weights_heron_ebit_duration,
        "ionq_forte": get_weights_ionq_forte_ebit_duration,
    }

    hw_labels = ["neutral_atom", "heron_ebit", "ionq_forte"]
    approaches = ["iterative", "alternating", "regular"]

    hw_labels_mono = ["neutral_atom", "heron", "ionq_forte"]
    global color_lst
    color_lst = color_lst_std

    # approach = "alternating"
    # hw_label = "neutral_atom"
    # ebit_time = 25000000

    # qrisp_circ_stats_plots_combined(x_vals=x_vals)

    # mono_plots(
    #     hw_labels_mono=hw_labels_mono,
    #     result_path_short=result_path_short,
    #     circ_names=circ_names,
    #     approaches=approaches,
    #     x_vals=x_vals,
    # )

    approaches_mono = [
        "iterative",
        "alternating",
        "regular",
    ]

    approaches_dist = [
        "iterative",
        "alternating",
        "regular",
    ]
    hw_labels_lim = ["neutral_atom"]
    hw_labels_all_app = ["neutral_atom", "heron_ebit", "ionq_forte"]
    color_lst = color_list_new

    # limited approaches
    dist_fixed_ebit_plots(
        hw_labels=hw_labels_lim,
        ebit_times_dict=ebit_times_dict,
        approaches=approaches_dist,
        result_path_short=result_path_short,
        circ_names=circ_names,
        x_vals=x_vals,
        EJPP_list=EJPP_list,
    )

    dist_fixed_N_plots(
        hw_labels=hw_labels_lim,
        ebit_times_dict=ebit_times_dict,
        circ_names=circ_names,
        approaches=approaches_dist,
        EJPP_list=EJPP_list,
        result_path_short=result_path_short,
        use_ms=True,
        plot_thresholds=False,
    )

    # # all approaches
    # dist_fixed_ebit_plots(
    #     hw_labels=hw_labels_all_app,
    #     ebit_times_dict=ebit_times_dict,
    #     approaches=approaches_mono,
    #     result_path_short=result_path_short,
    #     circ_names=circ_names,
    #     x_vals=x_vals,
    #     EJPP_list=EJPP_list,
    # )

    # dist_fixed_N_plots(
    #     hw_labels=hw_labels_all_app,
    #     ebit_times_dict=ebit_times_dict,
    #     circ_names=circ_names,
    #     approaches=approaches_mono,
    #     EJPP_list=EJPP_list,
    #     result_path_short=result_path_short,
    #     use_ms=True,
    #     plot_thresholds=False,
    # )

    # for hw_label in hw_labels:
    #     ebit_times = ebit_times_dict[hw_label]
    #     for approach in approaches:
    #         run_heatmap_plot(
    #             result_path_root=result_path_short,
    #             x_vals=x_vals,
    #             ebit_times=ebit_times,
    #             EJPP_list=EJPP_list,
    #             hw_label=hw_label,
    #             approach=approach,
    # )
    x_vals_heatmap = [
        1311,
        3111,
        7111,
        13111,
        31111,
        41111,
        71111,
        141111,
        411111,
        711111,
        1311111,
        13111111,
        141111111,
        1411111111,
        13111111111,
        131111111111,
        711111111111,
        33111111111111,
        711111111111111,
        711111111111111111,
        13111111111111111111,
    ]

    global cmap
    cmap = cmap_new

    # for hw_label in hw_labels_mono:
    #     run_heatmap_plot_bound(
    #         result_path_root=result_path_short,
    #         x_vals=x_vals_heatmap,
    #         ebit_times=ebit_times_dict[hw_label],
    #         weight_func=weight_func_dict[hw_label],
    #         hw_label=hw_label,
    #         dlog=False,
    #     )

    # run_heatmap_plot_combined_approach(
    #     result_path_root=result_path_short,
    #     x_vals=x_vals_heatmap,
    #     ebit_times=ebit_times_dict["neutral_atom"],
    #     EJPP_list=EJPP_list,
    #     hw_label="neutral_atom",
    #     approach_list=approaches,
    #     margin_scale=True,
    # )
    # run_heatmap_plot_combined_hw(
    #     result_path_root=result_path_short,
    #     x_vals=x_vals_heatmap,
    #     ebit_times_dict=ebit_times_dict,
    #     EJPP_list=EJPP_list,
    #     hw_list=hw_labels,
    #     approach="alternating",
    #     margin_scale=True,
    # )


def dlog_shor_plots():

    circ_names = [
        "N15_QRISP",
        "N21_QRISP",
        "N25_QRISP",
        "N33_QRISP",
        "N35_QRISP",
        "N39_QRISP",
        "N45_QRISP",
        "N49_QRISP",
        "N71_QRISP",
        "N711_QRISP",
        "N1311_QRISP",
        "N3111_QRISP",
        "N7111_QRISP",
        "N13111_QRISP",
        "N31111_QRISP",
        "N41111_QRISP",
        "N71111_QRISP",
        "N141111_QRISP",
        "N411111_QRISP",
        "N711111_QRISP",
        "N1311111_QRISP",
        "N13111111_QRISP",
        "N141111111_QRISP",
        "N1411111111_QRISP",
        "N13111111111_QRISP",
        "N131111111111_QRISP",
        "N711111111111_QRISP",
        "N33111111111111_QRISP",
        "N711111111111111_QRISP",
        "N711111111111111111_QRISP",
        "N13111111111111111111_QRISP",
    ]

    x_vals = [
        15,
        21,
        25,
        33,
        35,
        39,
        45,
        49,
        71,
        711,
        1311,
        3111,
        7111,
        13111,
        31111,
        41111,
        71111,
        141111,
        411111,
        711111,
        1311111,
        13111111,
        141111111,
        1411111111,
        13111111111,
        131111111111,
        711111111111,
        33111111111111,
        711111111111111,
        711111111111111111,
        13111111111111111111,
    ]

    EJPP_list = [i for i in range(1, 5)] + [6, 8, 10]

    ebit_list_ion = [
        (ebit_d_ms * 10**6) for ebit_d_ms in range(5, 20000, 500)
    ]  # 5ms to 20s

    ebit_list_sc = [(ebit_d_mus * 10**3) for ebit_d_mus in range(10, 1000, 20)]
    ebit_list_natom = [(ebit_d_ms * 10**6) for ebit_d_ms in range(5, 120, 5)]

    result_path_short = "./src/semi_iterative_comparison/results_dlog/"
    ebit_times_dict = {
        "dlog_neutral_atom": ebit_list_natom,
        "dlog_heron_ebit": ebit_list_sc,
        "dlog_ionq_forte": ebit_list_ion,
        "dlog_heron": ebit_list_sc,
    }

    hw_labels = ["dlog_heron_ebit", "dlog_ionq_forte"]
    hw_labels_all_app = ["dlog_neutral_atom"]
    approaches_mono = [
        "iterative",
        "double_iterative",
        "alternating",
        "three_cyclic",
        "regular",
    ]

    approaches_dist = [
        # "iterative",
        # "double_iterative",
        "alternating",
        # "three_cyclic",
        # "regular",
    ]

    weight_func_dict = {
        "dlog_neutral_atom": get_weights_neutral_atom_ebit_duration,
        "dlog_heron": get_weights_heron_ebit_duration,
        "dlog_ionq_forte": get_weights_ionq_forte_ebit_duration,
    }
    hw_labels_mono = ["dlog_neutral_atom", "dlog_heron", "dlog_ionq_forte"]

    global color_lst
    color_lst = color_lst_dlog

    # approach = "alternating"
    # hw_label = "neutral_atom"
    # ebit_time = 25000000

    # qrisp_circ_stats_plots_combined(x_vals=x_vals)

    # mono_plots(
    #     hw_labels_mono=hw_labels_mono,
    #     result_path_short=result_path_short,
    #     circ_names=circ_names,
    #     approaches=approaches_mono,
    #     x_vals=x_vals,
    # )

    color_lst = color_list_new

    # # limited approaches
    # dist_fixed_ebit_plots(
    #     hw_labels=hw_labels,
    #     ebit_times_dict=ebit_times_dict,
    #     approaches=approaches_dist,
    #     result_path_short=result_path_short,
    #     circ_names=circ_names,
    #     x_vals=x_vals,
    #     EJPP_list=EJPP_list,
    # )

    # dist_fixed_N_plots(
    #     hw_labels=hw_labels,
    #     ebit_times_dict=ebit_times_dict,
    #     circ_names=circ_names,
    #     approaches=approaches_dist,
    #     EJPP_list=EJPP_list,
    #     result_path_short=result_path_short,
    #     use_ms=True,
    #     plot_thresholds=False,
    # )

    # # all approaches
    # dist_fixed_ebit_plots(
    #     hw_labels=hw_labels_all_app,
    #     ebit_times_dict=ebit_times_dict,
    #     approaches=approaches_mono,
    #     result_path_short=result_path_short,
    #     circ_names=circ_names,
    #     x_vals=x_vals,
    #     EJPP_list=EJPP_list,
    # )

    # dist_fixed_N_plots(
    #     hw_labels=hw_labels_all_app,
    #     ebit_times_dict=ebit_times_dict,
    #     circ_names=circ_names,
    #     approaches=approaches_mono,
    #     EJPP_list=EJPP_list,
    #     result_path_short=result_path_short,
    #     use_ms=True,
    #     plot_thresholds=False,
    # )

    x_vals_heatmap = [
        1311,
        3111,
        7111,
        13111,
        31111,
        41111,
        71111,
        141111,
        411111,
        711111,
        1311111,
        13111111,
        141111111,
        1411111111,
        13111111111,
        131111111111,
        711111111111,
        33111111111111,
        711111111111111,
        711111111111111111,
        13111111111111111111,
    ]

    global cmap
    cmap = cmap_new

    for hw_label in hw_labels_mono:
        run_heatmap_plot_bound(
            result_path_root=result_path_short,
            x_vals=x_vals_heatmap,
            ebit_times=ebit_times_dict[hw_label],
            weight_func=weight_func_dict[hw_label],
            hw_label=hw_label,
            dlog=True,
        )

    for hw_label in hw_labels:
        ebit_times = ebit_times_dict[hw_label]
        for approach in ["alternating"]:
            run_heatmap_plot(
                result_path_root=result_path_short,
                x_vals=x_vals_heatmap,
                ebit_times=ebit_times,
                EJPP_list=EJPP_list,
                hw_label=hw_label,
                approach=approach,
                margin_scale=True,
            )

    for hw_label in hw_labels_all_app:
        ebit_times = ebit_times_dict[hw_label]
        for approach in approaches_mono:
            run_heatmap_plot(
                result_path_root=result_path_short,
                x_vals=x_vals_heatmap,
                ebit_times=ebit_times,
                EJPP_list=EJPP_list,
                hw_label=hw_label,
                approach=approach,
                margin_scale=True,
            )

    approaches_hm_app = [
        "iterative",
        # "double_iterative",
        "alternating",
        # "three_cyclic",
        "regular",
    ]

    run_heatmap_plot_combined_approach(
        result_path_root=result_path_short,
        x_vals=x_vals_heatmap,
        ebit_times=ebit_times_dict["dlog_neutral_atom"],
        EJPP_list=EJPP_list,
        hw_label="dlog_neutral_atom",
        approach_list=approaches_hm_app,
        margin_scale=True,
    )
    run_heatmap_plot_combined_hw(
        result_path_root=result_path_short,
        x_vals=x_vals_heatmap,
        ebit_times_dict=ebit_times_dict,
        EJPP_list=EJPP_list,
        hw_list=hw_labels_all_app + hw_labels,
        approach="alternating",
        margin_scale=True,
    )


if __name__ == "__main__":

    # dlog_shor_plots()
    default_shor_plots()
