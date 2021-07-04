from semantic_scholar_quantitative_data_common import analyse_file

fig, (ax3, ax1, ax2) = plt.subplots(3, 1, figsize=(7.5, 8.275), gridspec_kw={
    "hspace": 0.5
})

for key, value in analyse_file(
        utils.datafile(
            "semantic_scholar_open_corpus_2021_04_nef_citations.json")).items():
    globals()[key] = value


ax3.bar(years,
        y0,
        color=utils.oranges[0],
        label="First author affiliated with the CNRG ($N= {}$)".format(n0))
ax3.bar(years,
        y1 - y0,
        bottom=y0,
        color=utils.blues[1],
        label="At least one author affiliated with the CNRG ($N = {}$)".format(n1 - n0))
ax3.bar(years,
        y2 - y1,
        bottom=y1,
        color=utils.blues[0],
        label="No CNRG affiliation ($N = {}$)".format(n2 - n1))
for i, n in enumerate(y2):
    if y0[i] >= 10:
        ax3.text(years[i],
                 y0[i] * 0.5,
                 str(y0[i]) + ("*" if i + 1 == len(y2) else ""),
                 ha="center",
                 va="center",
                 color="white")
    if y1[i] - y0[i] >= 10:
        ax3.text(years[i], (y0[i] + y1[i]) * 0.5,
                 str(y1[i] - y0[i]) + ("*" if i + 1 == len(y2) else ""),
                 ha="center",
                 va="center",
                 color="white")
    if y2[i] - y1[i] >= 10:
        ax3.text(years[i], (y1[i] + y2[i]) * 0.5,
                 str(y2[i] - y1[i]) + ("*" if i + 1 == len(y2) else ""),
                 ha="center",
                 va="center",
                 color="white")
ax3.set_xticks(list(range(year_min, year_max, 2)))
ax3.set_xticks(years, minor=True)
ax3.set_yticks(np.arange(0, 151, 10), minor=True)
ax3.set_yticks(np.arange(0, 151, 50))
ax3.set_ylim(0, 150)
ax3.set_xlim(2003, 2022)
ax3.set_xlabel("Year")
ax3.set_ylabel("Publications per year")
ax3.text(-0.07, 1.095, "\\textbf{A}", size=12, ha="left", va="bottom", transform=ax3.transAxes)
ax3.set_title("\\textbf{Publications citing key NEF work by affiliation}", y=1.0475, va="bottom")
ax3.legend(loc="upper left")



for key, value in analyse_file(
        utils.datafile(
            "semantic_scholar_open_corpus_2021_04_nef_nengo.json")).items():
    globals()[key] = value

ax1.bar(years,
        y0,
        color=utils.oranges[0],
        label="First author affiliated with the CNRG ($N= {}$)".format(n0))
ax1.bar(years,
        y1 - y0,
        bottom=y0,
        color=utils.blues[1],
        label="At least one author affiliated with the CNRG ($N = {}$)".format(n1 - n0))
ax1.bar(years,
        y2 - y1,
        bottom=y1,
        color=utils.blues[0],
        label="No CNRG affiliation ($N = {}$)".format(n2 - n1))
for i, n in enumerate(y2):
    if y0[i] > 1:
        ax1.text(years[i],
                 y0[i] * 0.5,
                 str(y0[i]) + ("*" if i + 1 == len(y2) else ""),
                 ha="center",
                 va="center",
                 color="white")
    if y1[i] - y0[i] > 1:
        ax1.text(years[i], (y0[i] + y1[i]) * 0.5,
                 str(y1[i] - y0[i]) + ("*" if i + 1 == len(y2) else ""),
                 ha="center",
                 va="center",
                 color="white")
    if y2[i] - y1[i] > 1:
        ax1.text(years[i], (y1[i] + y2[i]) * 0.5,
                 str(y2[i] - y1[i]) + ("*" if i + 1 == len(y2) else ""),
                 ha="center",
                 va="center",
                 color="white")
ax1.set_xticks(list(range(year_min, year_max, 2)))
ax1.set_xticks(years, minor=True)
ax1.set_yticks(np.arange(0, 21), minor=True)
ax1.set_yticks(np.arange(0, 21, 5))
ax1.set_ylim(0, 20)
ax1.set_xlim(2003, 2022)
ax1.set_xlabel("Year")
ax1.set_ylabel("Publications per year", labelpad=8)
ax1.text(-0.07, 1.095, "\\textbf{B}", size=12, ha="left", va="bottom", transform=ax1.transAxes)
ax1.set_title("\\textbf{Publications with NEF-related keywords in the title or abstract by affiliation}", y=1.0475, va="bottom")
ax1.legend(loc="upper left")

data = {
    "Neuromorphics": (z_hw, utils.purples[1], "white"),
    "Neurobiological models": (z_bio, utils.greens[0], "white"),
    "Theory": (z_th, utils.blues[2], "black"),
#    "Software": (z_soft, utils.purples[2]),
#    "Robotics": (z_rob, utils.purples[0]),
}

offs = np.zeros_like(y2)
for name, (zs, color, tcolor) in data.items():
    ax2.bar(years,
            zs,
            bottom=offs,
            color=color,
            label=name + " ($N = {}$)".format(np.sum(zs)))
    for i in range(len(zs)):
        incomplete = ("*" if i + 1 == len(y2) else "")
        if zs[i] > 1:
            ax2.text(years[i],
                     offs[i] + zs[i] * 0.5,
                     str(zs[i]) + incomplete,
                     ha="center",
                     va="center",
                     color=tcolor)
    offs += zs

ax2.bar(years,
        y2 - offs,
        bottom=offs,
        color=utils.grays[3],
        label="Other (e.g., software, robotics; $N = {}$)".format(np.sum(y2 - offs)))
for i in range(len(y2)):
    incomplete = ("*" if i + 1 == len(y2) else "")
    if (y2[i] - offs[i]) > 1:
        ax2.text(years[i],
                 (offs[i] + y2[i]) * 0.5,
                 str(y2[i] - offs[i]) + incomplete,
                 ha="center",
                 va="center",
                 color=tcolor)
ax2.set_xticks(list(range(year_min, year_max, 2)))
ax2.set_xticks(years, minor=True)
ax2.set_yticks(np.arange(0, 21), minor=True)
ax2.set_yticks(np.arange(0, 21, 5))
ax2.set_ylim(0, 20)
ax2.set_xlim(2003, 2022)
ax2.set_xlabel("Year")
ax2.set_ylabel("Publications per year", labelpad=8)
ax2.text(-0.07, 1.095, "\\textbf{C}", size=12, ha="left", va="bottom", transform=ax2.transAxes)
ax2.set_title("\\textbf{Publications with NEF-related keywords in the title or abstract by topic}", y=1.0475, va="bottom")

ax2.legend(loc="upper left")

utils.save(fig)

