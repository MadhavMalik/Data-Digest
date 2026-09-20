"""NYC TLC Trip Record Data source + the official data dictionary.

The data dictionary is the reason this engine can avoid the classic naive-LLM
failures.  Two facts in particular do real work downstream:

  * `tip_amount` is populated for CREDIT CARD tips.  Cash tips are not
    included.  So "cash riders tip less" is an unsupportable conclusion from
    this field, no matter how clean the correlation looks.

  * `total_amount` is the SUM of the other fare components.  Its correlation
    with `fare_amount` is an accounting identity, not a discovery.

Both are encoded as machine-readable flags (`caveats`, `accounting_identity`)
so the critic enforces them rather than relying on a prompt to remember.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from signal_engine.ingestion.base import (
    DatasetHandle,
    DownloadResult,
    fingerprint_file,
    iter_progress_bar,
    stream_download,
    validate_parquet,
)

TLC_CDN = "https://d37ci6vzurychx.cloudfront.net"
TLC_TRIP_DATA = f"{TLC_CDN}/trip-data"
TLC_ZONE_LOOKUP_URL = f"{TLC_CDN}/misc/taxi_zone_lookup.csv"
TLC_PAGE = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
VOLORIDGE_REGISTRY = "https://registry.opendata.aws/nyc-tlc-trip-records-pds/"


class TLCVehicle(str, Enum):
    YELLOW = "yellow"
    GREEN = "green"
    FHV = "fhv"
    FHVHV = "fhvhv"

    @property
    def file_prefix(self) -> str:
        return {
            TLCVehicle.YELLOW: "yellow_tripdata",
            TLCVehicle.GREEN: "green_tripdata",
            TLCVehicle.FHV: "fhv_tripdata",
            TLCVehicle.FHVHV: "fhvhv_tripdata",
        }[self]


@dataclass(frozen=True)
class TLCSource:
    """Resolves one monthly TLC Parquet file.

    Deliberately parameterized rather than a hardcoded URL: the same code path
    serves yellow/green/fhvhv for any month without edits.
    """

    vehicle: TLCVehicle = TLCVehicle.YELLOW
    year: int = 2026
    month: int = 1

    def __post_init__(self) -> None:
        if not 1 <= self.month <= 12:
            raise ValueError(f"month must be 1-12, got {self.month}")
        if not 2009 <= self.year <= 2100:
            raise ValueError(f"implausible year: {self.year}")

    def local_filename(self) -> str:
        return f"{self.vehicle.file_prefix}_{self.year:04d}-{self.month:02d}.parquet"

    def resolve_url(self) -> str:
        return f"{TLC_TRIP_DATA}/{self.local_filename()}"

    def dataset_id(self) -> str:
        return f"nyc_tlc_{self.vehicle.value}_{self.year:04d}_{self.month:02d}"

    # ---- fetch ----------------------------------------------------------
    def fetch(
        self, output_dir: Path, *, force: bool = False, show_progress: bool = True
    ) -> tuple[DatasetHandle, DownloadResult]:
        url = self.resolve_url()
        dest = Path(output_dir) / self.local_filename()
        progress = iter_progress_bar(self.local_filename()) if show_progress else None

        result = stream_download(url, dest, force=force, progress=progress)
        if show_progress:
            print()

        info = validate_parquet(dest)  # raises if the footer is bad

        handle = DatasetHandle(
            dataset_id=self.dataset_id(),
            path=dest,
            fingerprint=fingerprint_file(dest),
            row_count=info["row_count"],
            byte_size=info["byte_size"],
            source_url=url,
            description=(
                f"NYC TLC {self.vehicle.value} taxi trip records, "
                f"{self.year:04d}-{self.month:02d}. Official monthly Parquet release."
            ),
            extras={
                "vehicle": self.vehicle.value,
                "year": self.year,
                "month": self.month,
                "row_groups": info["row_groups"],
                "official_page": TLC_PAGE,
                "aws_registry": VOLORIDGE_REGISTRY,
            },
        )
        return handle, result


def fetch_zone_lookup(output_dir: Path, *, force: bool = False) -> Path:
    """Download the taxi-zone lookup table (zone id -> borough / zone name)."""
    dest = Path(output_dir) / "taxi_zone_lookup.csv"
    stream_download(TLC_ZONE_LOOKUP_URL, dest, force=force)
    return dest


# ===========================================================================
# Official Yellow Taxi data dictionary
# Source: NYC TLC "Data Dictionary - Yellow Taxi Trip Records"
# ===========================================================================

YELLOW_TAXI_DICTIONARY: dict[str, dict] = {
    "VendorID": {
        "description": "A code indicating the TPEP provider that supplied the record.",
        "semantic_type": "categorical",
        "unit": "category",
        "categories": {
            1: "Creative Mobile Technologies, LLC",
            2: "Curb Mobility, LLC",
            6: "Myle Technologies Inc",
            7: "Helix",
        },
        "is_identifier_like": True,
    },
    "tpep_pickup_datetime": {
        "description": "The date and time when the meter was engaged.",
        "semantic_type": "datetime",
        "unit": "timestamp",
    },
    "tpep_dropoff_datetime": {
        "description": "The date and time when the meter was disengaged.",
        "semantic_type": "datetime",
        "unit": "timestamp",
    },
    "passenger_count": {
        "description": "The number of passengers in the vehicle. This is a driver-entered value.",
        "semantic_type": "count",
        "unit": "count",
        "caveats": [
            "Driver-entered, not measured. Missing and zero values are common and do not "
            "necessarily mean the trip had no passengers."
        ],
    },
    "trip_distance": {
        "description": "The elapsed trip distance in miles reported by the taximeter.",
        "semantic_type": "continuous_measurement",
        "unit": "miles",
    },
    "RatecodeID": {
        "description": "The final rate code in effect at the end of the trip.",
        "semantic_type": "categorical",
        "unit": "category",
        "categories": {
            1: "Standard rate",
            2: "JFK",
            3: "Newark",
            4: "Nassau or Westchester",
            5: "Negotiated fare",
            6: "Group ride",
            99: "Null/unknown",
        },
        "is_identifier_like": True,
        "caveats": [
            "Rate codes 2 and 3 are flat/administered fares, so distance does not drive "
            "fare the same way it does for standard-rate trips."
        ],
    },
    "store_and_fwd_flag": {
        "description": (
            "Whether the trip record was held in vehicle memory before sending to the vendor "
            "because the vehicle had no connection to the server. Y=store and forward, N=not."
        ),
        "semantic_type": "boolean",
        "unit": "bool",
    },
    "PULocationID": {
        "description": "TLC Taxi Zone in which the taximeter was engaged.",
        "semantic_type": "categorical_identifier",
        "unit": "id",
        "is_identifier_like": True,
        "caveats": [
            "Zone IDs are arbitrary labels. Their numeric magnitude and ordering carry no "
            "meaning, so correlation with a zone ID is not interpretable as a trend."
        ],
    },
    "DOLocationID": {
        "description": "TLC Taxi Zone in which the taximeter was disengaged.",
        "semantic_type": "categorical_identifier",
        "unit": "id",
        "is_identifier_like": True,
        "caveats": [
            "Zone IDs are arbitrary labels. Their numeric magnitude and ordering carry no "
            "meaning, so correlation with a zone ID is not interpretable as a trend."
        ],
    },
    "payment_type": {
        "description": "A numeric code signifying how the passenger paid for the trip.",
        "semantic_type": "categorical",
        "unit": "category",
        "categories": {
            0: "Flex Fare trip",
            1: "Credit card",
            2: "Cash",
            3: "No charge",
            4: "Dispute",
            5: "Unknown",
            6: "Voided trip",
        },
        "is_identifier_like": True,
    },
    "fare_amount": {
        "description": "The time-and-distance fare calculated by the meter.",
        "semantic_type": "currency",
        "unit": "USD",
        "fare_component": True,
    },
    "extra": {
        "description": (
            "Miscellaneous extras and surcharges. Currently this only includes the $0.50 and "
            "$1.00 rush hour and overnight charges."
        ),
        "semantic_type": "currency",
        "unit": "USD",
        "fare_component": True,
    },
    "mta_tax": {
        "description": "$0.50 MTA tax that is automatically triggered based on the metered rate in use.",
        "semantic_type": "currency",
        "unit": "USD",
        "fare_component": True,
    },
    "tip_amount": {
        "description": (
            "Tip amount. This field is automatically populated for credit card tips. "
            "Cash tips are not included."
        ),
        "semantic_type": "currency",
        "unit": "USD",
        "fare_component": True,
        "caveats": [
            "CRITICAL: cash tips are NOT recorded in this field. Any comparison of tipping "
            "behaviour across payment_type values is measuring what the system records, not "
            "what passengers actually tipped. Cash trips will show near-zero tip_amount by "
            "construction."
        ],
        "blocks_claims": [
            {
                "involves_columns": ["payment_type"],
                "forbidden_claim": "cash passengers tip less than card passengers",
                "reason": (
                    "tip_amount only captures credit-card tips; cash tips are structurally "
                    "absent from the data, so the comparison is undefined."
                ),
            }
        ],
    },
    "tolls_amount": {
        "description": "Total amount of all tolls paid in trip.",
        "semantic_type": "currency",
        "unit": "USD",
        "fare_component": True,
    },
    "improvement_surcharge": {
        "description": (
            "$0.30 improvement surcharge assessed on trips at the flag drop. The improvement "
            "surcharge began being levied in 2015."
        ),
        "semantic_type": "currency",
        "unit": "USD",
        "fare_component": True,
    },
    "total_amount": {
        "description": "The total amount charged to passengers. Does not include cash tips.",
        "semantic_type": "currency",
        "unit": "USD",
        "caveats": [
            "total_amount is the SUM of the other fare components. Correlation between "
            "total_amount and any component is a mechanical accounting relationship."
        ],
    },
    "congestion_surcharge": {
        "description": "Total amount collected in trip for NYS congestion surcharge.",
        "semantic_type": "currency",
        "unit": "USD",
        "fare_component": True,
    },
    "airport_fee": {
        "description": "For pick up only at LaGuardia and John F. Kennedy Airports.",
        "semantic_type": "currency",
        "unit": "USD",
        "fare_component": True,
        "caveats": [
            "A fixed per-pickup fee at LGA/JFK only. It is added regardless of distance, so it "
            "inflates cost-per-mile on airport pickups independently of trip length."
        ],
    },
    "cbd_congestion_fee": {
        "description": (
            "Per-trip charge for MTA's Congestion Relief Zone, which started being levied on "
            "January 5, 2025."
        ),
        "semantic_type": "currency",
        "unit": "USD",
        "fare_component": True,
    },
    "Airport_fee": {  # some monthly files use this capitalization
        "description": "For pick up only at LaGuardia and John F. Kennedy Airports.",
        "semantic_type": "currency",
        "unit": "USD",
        "fare_component": True,
        "alias_of": "airport_fee",
    },
}


# `total_amount` is definitionally the sum of these.  The critic uses this to
# label such relationships as mechanical rather than discovered.
YELLOW_ACCOUNTING_IDENTITIES: list[dict] = [
    {
        "target": "total_amount",
        "components": [
            "fare_amount",
            "extra",
            "mta_tax",
            "tip_amount",
            "tolls_amount",
            "improvement_surcharge",
            "congestion_surcharge",
            "airport_fee",
            "cbd_congestion_fee",
        ],
        "relation": "sum",
        "note": (
            "total_amount = sum of fare components (excluding cash tips). Any strong "
            "correlation with a component is an accounting identity."
        ),
    }
]


YELLOW_DATASET_NOTES = [
    "Each row is one completed taxi trip as reported by a TPEP provider.",
    "Records are submitted by technology providers; TLC does not guarantee accuracy.",
    "Trips with RatecodeID 2 (JFK) and 3 (Newark) use flat or administered fares.",
    "Negative fare amounts appear in the raw files and usually indicate refunds or voided trips.",
    "A small number of records carry timestamps outside the nominal month of the file.",
]


def write_dictionary_metadata(metadata_dir: Path) -> Path:
    """Persist the dictionary as machine-readable JSON under data/metadata/."""
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    dest = metadata_dir / "nyc_tlc_yellow_dictionary.json"
    payload = {
        "dataset": "NYC TLC Yellow Taxi Trip Records",
        "source_page": TLC_PAGE,
        "aws_registry": VOLORIDGE_REGISTRY,
        "columns": YELLOW_TAXI_DICTIONARY,
        "accounting_identities": YELLOW_ACCOUNTING_IDENTITIES,
        "dataset_notes": YELLOW_DATASET_NOTES,
    }
    dest.write_text(json.dumps(payload, indent=2, sort_keys=False))
    return dest


def dictionary_for(vehicle: TLCVehicle) -> dict[str, dict]:
    if vehicle is TLCVehicle.YELLOW:
        return YELLOW_TAXI_DICTIONARY
    # Green shares most yellow columns (with lpep_* timestamps).
    if vehicle is TLCVehicle.GREEN:
        green = dict(YELLOW_TAXI_DICTIONARY)
        green["lpep_pickup_datetime"] = YELLOW_TAXI_DICTIONARY["tpep_pickup_datetime"]
        green["lpep_dropoff_datetime"] = YELLOW_TAXI_DICTIONARY["tpep_dropoff_datetime"]
        green["trip_type"] = {
            "description": "A code indicating whether the trip was a street-hail or a dispatch.",
            "semantic_type": "categorical",
            "unit": "category",
            "categories": {1: "Street-hail", 2: "Dispatch"},
            "is_identifier_like": True,
        }
        return green
    return {}


def accounting_identities_for(vehicle: TLCVehicle) -> list[dict]:
    if vehicle in (TLCVehicle.YELLOW, TLCVehicle.GREEN):
        return YELLOW_ACCOUNTING_IDENTITIES
    return []
