"""Convert Hill of Towie fast-log records to causal one-minute Parquet.

The source archive stores one Parquet stream per turbine, signal, and UTC day.
For compatibility with the archived manuscript inputs, each stream is sampled
on the integer-second clock by last-observation-carried-forward (LOCF).  The
60 causal second values are then averaged within each minute; minute minimum
and maximum power are calculated from the same one-second power series.

No interpolation from a future observation is used.  A day is emitted only
when every required signal has a source member, so a fully missing turbine-day
is retained as a data gap rather than being filled across 24 hours.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = PROJECT_DIR / "data" / "hill_of_towie" / "turbine_fastlog.zip"
DEFAULT_OUTPUT = PROJECT_DIR / "data" / "hill_of_towie" / "converted"
STATION_OFFSET = 2_304_509

# Ordered to reproduce the archived column contract exactly.
SIGNALS = OrderedDict(
    (
        ("Wtc_TDI_AcWindSp_AcWindSp", "AcWindSp_AcWindSp"),
        ("Wtc_TDI_ActLimit_Power", "ActLimit_Power"),
        ("Wtc_TDI_ActPower_Value", "ActPower_Value"),
        ("Wtc_TDI_MainSRpm_Value", "GenRpm_Value"),
        ("Wtc_TDI_PitcPosA_Value", "PitcPosA_Value"),
        ("Wtc_TDI_PowerRed_PowerRed", "PowerRed_PowerRed"),
        ("Wtc_TDI_PowerRef_PowerRef", "PowerRef_PowerRef"),
    )
)
DEFAULT_TURBINES = ("T01", "T02", "T03", "T04", "T05", "T07", "T11", "T13", "T14")
MEMBER_RE = re.compile(
    r"/HOT/(?P<station>\d+)/(?P<date>\d{4}-\d{2}-\d{2})/"
    r"FL(?P=station)_(?P<signal>.+)_(?P<file_date>\d{4}_\d{2}_\d{2})\.prq$"
)


def turbine_station(turbine: str) -> str:
    """Map a label such as ``T11`` to its Hill of Towie station identifier."""

    match = re.fullmatch(r"T(\d{1,2})", turbine.upper())
    if match is None:
        raise ValueError("invalid turbine label %r; expected T01--T22" % turbine)
    number = int(match.group(1))
    if not 1 <= number <= 22:
        raise ValueError("turbine number is outside the published T01--T22 range")
    return str(STATION_OFFSET + number)


def discover_members(archive: zipfile.ZipFile) -> Dict[Tuple[str, str, str], str]:
    """Index required members by ``(station, YYYY-MM-DD, raw signal)``."""

    members: Dict[Tuple[str, str, str], str] = {}
    required = set(SIGNALS)
    for name in archive.namelist():
        match = MEMBER_RE.search(name)
        if match is None or match.group("signal") not in required:
            continue
        key = (match.group("station"), match.group("date"), match.group("signal"))
        if key in members:
            raise RuntimeError("duplicate fast-log member for %r" % (key,))
        members[key] = name
    return members


def read_member_series(archive: zipfile.ZipFile, member: str) -> pd.Series:
    """Read one compressed source member without extracting the archive."""

    with archive.open(member) as handle:
        table = pq.read_table(io.BytesIO(handle.read()))
    frame = table.to_pandas()
    if frame.shape[1] != 1 or not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("unexpected fast-log schema in %s" % member)
    series = frame.iloc[:, 0].astype(float).sort_index()
    if series.index.has_duplicates:
        series = series.groupby(level=0, sort=True).last()
    return series


def causal_second_grid(
    series: pd.Series,
    day: pd.Timestamp,
    previous_value: Optional[float] = None,
) -> Tuple[pd.Series, Optional[float]]:
    """Sample one UTC day on integer seconds using only past observations."""

    start = pd.Timestamp(day).normalize()
    stop = start + pd.Timedelta(days=1)
    current = series.loc[(series.index >= start) & (series.index < stop)]
    pieces = []
    if previous_value is not None and np.isfinite(previous_value):
        pieces.append(
            pd.Series([previous_value], index=[start - pd.Timedelta(nanoseconds=1)])
        )
    pieces.append(current)
    source = pd.concat(pieces).sort_index()
    grid = pd.date_range(start, periods=86_400, freq="1s")
    sampled = source.reindex(grid, method="ffill")
    next_value = previous_value
    if not current.empty and np.isfinite(current.iloc[-1]):
        next_value = float(current.iloc[-1])
    return sampled.astype(float), next_value


def minute_frame_for_day(
    archive: zipfile.ZipFile,
    members: Mapping[Tuple[str, str, str], str],
    station: str,
    day: pd.Timestamp,
    carry: Dict[str, Optional[float]],
) -> pd.DataFrame:
    """Convert one complete station-day to the archived nine-column contract."""

    date_text = day.strftime("%Y-%m-%d")
    seconds: Dict[str, pd.Series] = {}
    for raw_signal in SIGNALS:
        member = members[(station, date_text, raw_signal)]
        raw = read_member_series(archive, member)
        seconds[raw_signal], carry[raw_signal] = causal_second_grid(
            raw, day, carry.get(raw_signal)
        )

    power = seconds["Wtc_TDI_ActPower_Value"]
    minute = pd.DataFrame(
        {
            "AcWindSp_AcWindSp": seconds["Wtc_TDI_AcWindSp_AcWindSp"].resample("1min").mean(),
            "ActLimit_Power": seconds["Wtc_TDI_ActLimit_Power"].resample("1min").mean(),
            "ActPower_Value": power.resample("1min").mean(),
            "min_ActPower_Value": power.resample("1min").min(),
            "max_ActPower_Value": power.resample("1min").max(),
            "GenRpm_Value": seconds["Wtc_TDI_MainSRpm_Value"].resample("1min").mean(),
            "PitcPosA_Value": seconds["Wtc_TDI_PitcPosA_Value"].resample("1min").mean(),
            "PowerRed_PowerRed": seconds["Wtc_TDI_PowerRed_PowerRed"].resample("1min").mean(),
            "PowerRef_PowerRef": seconds["Wtc_TDI_PowerRef_PowerRef"].resample("1min").mean(),
        }
    )
    minute.index.name = None
    return minute


def available_dates(
    members: Mapping[Tuple[str, str, str], str], station: str
) -> list[pd.Timestamp]:
    """Return dates for which all seven required source signals are present."""

    by_signal = {
        signal: {
            date for candidate_station, date, candidate_signal in members
            if candidate_station == station and candidate_signal == signal
        }
        for signal in SIGNALS
    }
    complete = set.intersection(*(dates for dates in by_signal.values()))
    return [pd.Timestamp(date) for date in sorted(complete)]


def convert_turbine(
    archive: zipfile.ZipFile,
    members: Mapping[Tuple[str, str, str], str],
    turbine: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Convert one turbine, preserving fully missing days as timestamp gaps."""

    station = turbine_station(turbine)
    dates = available_dates(members, station)
    if start_date is not None:
        dates = [date for date in dates if date >= pd.Timestamp(start_date)]
    if end_date is not None:
        dates = [date for date in dates if date <= pd.Timestamp(end_date)]
    if not dates:
        raise ValueError("no complete fast-log days found for %s" % turbine)

    carry: Dict[str, Optional[float]] = {signal: None for signal in SIGNALS}
    daily = []
    previous_day: Optional[pd.Timestamp] = None
    for day in dates:
        # Never carry a value across a fully missing UTC day.
        if previous_day is not None and day - previous_day != pd.Timedelta(days=1):
            carry = {signal: None for signal in SIGNALS}
        daily.append(minute_frame_for_day(archive, members, station, day, carry))
        previous_day = day

    frame = pd.concat(daily).sort_index()
    frame.columns = pd.MultiIndex.from_product([[station], frame.columns])
    frame.columns.names = [None, None]
    frame.index.name = None
    return frame


def selftest() -> None:
    """Verify the causal sampling and minute aggregation contract."""

    index = pd.to_datetime(
        [
            "2026-01-01 00:00:00.000",
            "2026-01-01 00:00:00.500",
            "2026-01-01 00:00:01.200",
            "2026-01-01 00:00:59.900",
        ]
    )
    source = pd.Series([1.0, 99.0, 2.0, 3.0], index=index)
    sampled, carry = causal_second_grid(source, pd.Timestamp("2026-01-01"))
    assert sampled.iloc[0] == 1.0
    assert sampled.iloc[1] == 99.0
    assert sampled.iloc[2] == 2.0
    assert sampled.iloc[59] == 2.0
    assert sampled.iloc[60] == 3.0
    assert carry == 3.0
    assert sampled.iloc[:60].mean() == (1.0 + 99.0 + 58.0 * 2.0) / 60.0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--turbines", nargs="+", default=list(DEFAULT_TURBINES))
    parser.add_argument("--start-date", help="first UTC day, YYYY-MM-DD")
    parser.add_argument("--end-date", help="last UTC day, YYYY-MM-DD")
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    parser.add_argument("--list", action="store_true", help="list complete-day counts and exit")
    parser.add_argument(
        "--selftest", action="store_true", help="run the LOCF contract test and exit"
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.selftest:
        selftest()
        print("LOCF conversion self-test passed")
        return 0
    if not arguments.archive.exists():
        raise SystemExit("missing source archive: %s" % arguments.archive)

    with zipfile.ZipFile(arguments.archive) as archive:
        members = discover_members(archive)
        if arguments.list:
            for turbine in arguments.turbines:
                dates = available_dates(members, turbine_station(turbine))
                if dates:
                    print(
                        "%s: %d complete days (%s to %s)"
                        % (turbine, len(dates), dates[0].date(), dates[-1].date())
                    )
                else:
                    print("%s: no complete days" % turbine)
            return 0

        arguments.output_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for turbine in arguments.turbines:
            destination = arguments.output_dir / ("fl_df_HOT_%s_1min.parquet" % turbine.upper())
            if destination.exists() and not arguments.force:
                raise SystemExit("output exists (use --force): %s" % destination)
            frame = convert_turbine(
                archive,
                members,
                turbine.upper(),
                start_date=arguments.start_date,
                end_date=arguments.end_date,
            )
            frame.to_parquet(destination)
            outputs.append(
                {
                    "turbine": turbine.upper(),
                    "station": turbine_station(turbine),
                    "path": destination.name,
                    "rows": int(len(frame)),
                    "start": frame.index.min().isoformat(),
                    "end": frame.index.max().isoformat(),
                    "maximum_gap_seconds": float(
                        frame.index.to_series().diff().dt.total_seconds().max()
                    ),
                }
            )
            print("wrote %s (%d rows)" % (destination, len(frame)))

    manifest = {
        "source_archive": str(arguments.archive),
        "method": "integer-second causal LOCF, then one-minute mean/min/max",
        "required_signals": list(SIGNALS),
        "outputs": outputs,
    }
    (arguments.output_dir / "fastlog_conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
