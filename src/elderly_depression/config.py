"""Path and experiment configuration."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_CHARLS_ROOT = PROJECT_ROOT.parent / "中国健康与养老追踪调查数据库（2011-2020官网原始数据初步清洗后）"
_DEFAULT_MPDD_ROOT = PROJECT_ROOT / "data" / "mpdd"
_DEFAULT_CMDC_ROOT = PROJECT_ROOT.parent / "新数据集1"

CHARLS_ROOT = Path(os.getenv("CHARLS_ROOT", str(_DEFAULT_CHARLS_ROOT)))
MPDD_ROOT = Path(os.getenv("MPDD_ROOT", str(_DEFAULT_MPDD_ROOT)))
CMDC_ROOT = Path(os.getenv("CMDC_ROOT", str(_DEFAULT_CMDC_ROOT)))

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

HARMONIZED_GLOB = "Harmonized CHARLS/**/clean_H_CHARLS_D_Data.dta"

MIN_AGE = int(os.getenv("MIN_AGE", "60"))
CESD_THRESHOLD = 10
RANDOM_STATE = 42
TEST_SIZE = 0.2
