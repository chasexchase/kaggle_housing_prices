import shutil
from pathlib import Path
import kagglehub

path = Path(kagglehub.competition_download('house-prices-advanced-regression-techniques'))
dst = Path('raw')
dst.mkdir(parents=True, exist_ok=True)

for f in path.iterdir():
    shutil.copy(f, dst / f.name)

print(sorted(p.name for p in dst.iterdir()))