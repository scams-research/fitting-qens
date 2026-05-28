from pathlib import Path

# Absolute path to the top level of the repository
root = Path(__file__).resolve().parents[4].absolute()

# Absolute path to the `src` folder
src = root / "src"

# Absolute path to the `src/data` folder (contains datasets)
data = src / "data"

# Absolute path to the `src/scripts` folder (contains figure/pipeline scripts)
code = src / "code"
