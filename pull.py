from typing import Dict, List

import pandas as pd
import numpy as np
from collections import namedtuple
import json

from pull_codes import KEEP_CODES

DATA_URLS = {
    "PP": "https://www.casact.org/sites/default/files/2021-04/ppauto_pos.csv",
    "WC": "https://www.casact.org/sites/default/files/2021-04/wkcomp_pos.csv",
    "CA": "https://www.casact.org/sites/default/files/2021-04/comauto_pos.csv",
    "OO": "https://www.casact.org/sites/default/files/2021-04/othliab_pos.csv",
}

DATA_DIR = "data/"
ORIGIN_ACCIDENT_YEAR = 1988
T_DEV_TRAIN = 70
T_DEV_TEST = 20
T_DEV_VALID = 10
T_FORECAST_TRAIN = 9
T_FORECAST_TEST = 10 - T_FORECAST_TRAIN
ULTIMATE = 10

Cell = namedtuple(
    "Cell",
    (
        "code",
        "accident_year",
        "evaluation_year",
        "development_lag",
        "incurred_loss",
        "paid_loss",
        "earned_premium",
    ),
)
DataDictType = Dict[str, int | List[float] | List[int]]


def download_data(lob: str) -> pd.DataFrame:
    return pd.read_csv(DATA_URLS[lob])


def make_cells(data: pd.DataFrame, lob: str) -> Dict:
    cols = [
        "GRCODE",
        "AccidentYear",
        "DevelopmentYear",
        "DevelopmentLag",
        "IncurLoss",
        "CumPaidLoss",
        "EarnedPremDIR",
    ]
    cells = [
        Cell(*values) for _, values in data.filter(regex="|".join(cols)).iterrows()
    ]
    if KEEP_CODES[lob]:
        return [cell for cell in cells if cell.code in KEEP_CODES[lob]]
    return [cell for cell in cells if cell.code]


def build_development_data(cells: List[Cell]) -> DataDictType:
    raw = {}
    for cell in cells:
        if cell.code in raw:
            raw[cell.code].append(cell)
        else:
            raw[cell.code] = [cell]
    return {
        i: np.array(
            [
                (cell.paid_loss, cell.incurred_loss, cell.earned_premium)
                for cell in cells
            ]
        )
        .reshape((10, 10, 3))
        .tolist()
        for i, (_, cells) in enumerate(raw.items())
        if all(cell.paid_loss > 0 for cell in cells)
    }


def pull_lob_data(lob: str) -> None:
    data = download_data(lob)
    cells = make_cells(data, lob)
    development_data = build_development_data(cells)
    with open(DATA_DIR + f"{lob.lower()}.json", "w") as outfile:
        json.dump(development_data, outfile)
    return None


def main():
    for lob in DATA_URLS:
        print(f"Pulling {lob} data...")
        pull_lob_data(lob)


if __name__ == "__main__":
    main()
