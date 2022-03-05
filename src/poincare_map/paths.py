from pathlib import Path

# Directories
data = Path("data")
figures = Path("figures")
odes = Path("odes")

data.mkdir(exist_ok=True)
figures.mkdir(exist_ok=True)
odes.mkdir(exist_ok=True)

# Data files
numeric_bif_diagram = data / "numeric-bif-diagram.parquet"
analytic_bif_diagram = data / "analytic-bif-diagram.parquet"
