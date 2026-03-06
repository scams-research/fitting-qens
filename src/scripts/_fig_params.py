from matplotlib import rcParams
import matplotlib.font_manager as fm
helvetica = False
try:
    font_files = fm.findSystemFonts("/Users/ee23687/workspace/other/fonts", fontext="ttf")
    for font_file in font_files:
        fm.fontManager.addfont(font_file)
    helvetica = True
except Exception as e:
    print("Warning: Could not load Helvetica font.")
    print(f"{e}")
    print("Falling back to default sans-serif font.")


# blue, orange, green, pink, dark green, grey
colors = ["#0173B2", "#029E73", "#D55E00", "#CC78BC", "#ECE133", "#56B4E9"]
FONTSIZE = 6
NEARLY_BLACK = "#161616"
LIGHT_GREY = "#F5F5F5"
WHITE = "#ffffff"
NBINS = 25
ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
CREDIBLE_INTERVALS = [[15.9, 84.1, 0.6], [2.3, 97.7, 0.3], [0.15, 99.85, 0.1]]

MASTER_FORMATTING = {
    "axes.formatter.limits": (-3, 3),
    "xtick.major.pad": 1,
    "ytick.major.pad": 1,
    "ytick.color": NEARLY_BLACK,
    "xtick.color": NEARLY_BLACK,
    "axes.labelcolor": NEARLY_BLACK,
    "axes.spines.bottom": True,
    "axes.spines.left": True,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 1.0,
    "axes.axisbelow": True,
    "legend.frameon": False,
    "legend.handletextpad": 0.4,
    'axes.edgecolor': NEARLY_BLACK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # "mathtext.fontset": "stixsans",
    "mathtext.fontset": "custom",
    "mathtext.bf": "sans:bold",
    "mathtext.bfit": "sans:bold:italic",
    "mathtext.it": "sans:italic",
    "mathtext.rm": "sans",
    "mathtext.default": "it",
    "font.size": FONTSIZE,
    "font.family": "sans-serif",
    "savefig.bbox": "tight",
    "axes.facecolor": WHITE,
    "axes.labelpad": 1.0,
    "axes.labelsize": FONTSIZE,
    "axes.titlepad": 8,
    "axes.titlesize": FONTSIZE,
    "axes.grid": False,
    "grid.color": WHITE,
    "lines.markersize": 1.0,
    "xtick.major.size": 2.0,
    "xtick.major.width": 1.0,
    "ytick.major.size": 2.0,
    "ytick.major.width": 1.0,
    "lines.scale_dashes": False,
    "xtick.labelsize": FONTSIZE,
    "ytick.labelsize": FONTSIZE,
    "legend.fontsize": FONTSIZE,
    "lines.linewidth": 1,
}
if helvetica:
    MASTER_FORMATTING["font.family"] = "Helvetica"
    MASTER_FORMATTING["font.sans-serif"] = ["Helvetica"]

for k, v in MASTER_FORMATTING.items():
    rcParams[k] = v

# 510.0pt. \textwidth
# 246.0pt. \columnwidth

COLUMN_WIDTH = 246.0 / 72.27  # inches
PAGE_WIDTH = 510.0 / 72.27  # inches
GOLDEN_RATIO = (1 + 5**0.5) / 2  # Approx. 1.618