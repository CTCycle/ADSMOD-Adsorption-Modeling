from __future__ import annotations

import io
import os
from typing import Any

import pandas as pd

from ADSMOD.server.utils.configurations import server_settings
from ADSMOD.server.utils.constants import DATASET_FALLBACK_DELIMITERS
from ADSMOD.server.utils.repository.serializer import DataSerializer


###############################################################################
class DatasetService:
    def __init__(self) -> None:
        self.allowed_extensions = set(server_settings.datasets.allowed_extensions)

    # -------------------------------------------------------------------------
    def derive_dataset_name(self, filename: str | None) -> str:
        if isinstance(filename, str):
            base = os.path.splitext(os.path.basename(filename))[0].strip()
            if base:
                return base
        return "uploaded_dataset"

    # -------------------------------------------------------------------------
    def escape_markdown_table_cell(self, value: object) -> str:
        text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    # -------------------------------------------------------------------------
    def load_from_bytes(
        self, payload: bytes, filename: str | None
    ) -> tuple[dict[str, Any], str]:
        """Load an uploaded dataset payload and provide a serialized representation.

        Keyword arguments:
        payload -- Raw file bytes obtained from the upload endpoint.
        filename -- Original filename that hints at the file extension, if available.

        Return value:
        Tuple containing a JSON-serializable dataset description and a human-readable
        summary.
        """
        if not payload:
            raise ValueError("Uploaded dataset is empty.")

        dataframe = self.read_dataframe(payload, filename)
        serializable = dataframe.where(pd.notna(dataframe), None)
        dataset_name = self.derive_dataset_name(filename)
        dataset_payload: dict[str, Any] = {
            "dataset_name": dataset_name,
            "columns": list(serializable.columns),
            "records": serializable.to_dict(orient="records"),
            "row_count": int(serializable.shape[0]),
        }
        summary = self.format_dataset_summary(dataframe)
        return dataset_payload, summary

    # -------------------------------------------------------------------------
    def read_dataframe(self, payload: bytes, filename: str | None) -> pd.DataFrame:
        """Decode the uploaded file into a Pandas DataFrame, handling CSV and Excel inputs.

        Keyword arguments:
        payload -- Raw bytes representing the uploaded file contents.
        filename -- Provided filename used to infer the file format.

        Return value:
        DataFrame containing the parsed dataset ready for further processing.
        """
        extension = ""
        if isinstance(filename, str):
            extension = os.path.splitext(filename)[1].lower()

        if extension and extension not in self.allowed_extensions:
            raise ValueError(f"Unsupported file type: {extension}")

        buffer = io.BytesIO(payload)

        if extension in {".xls", ".xlsx"}:
            buffer.seek(0)
            dataframe = pd.read_excel(buffer, sheet_name=0)
        else:
            buffer.seek(0)
            dataframe = pd.read_csv(buffer, comment="#")

            if dataframe.shape[1] == 1:
                column_name = dataframe.columns[0]
                first_value = None
                if not dataframe.empty:
                    first_value = dataframe.iloc[0, 0]

                # When the parser reports a single column we attempt alternative
                # delimiters to handle semi-colon, tab, or pipe separated files.
                for delimiter in DATASET_FALLBACK_DELIMITERS:
                    if (isinstance(column_name, str) and delimiter in column_name) or (
                        isinstance(first_value, str) and delimiter in first_value
                    ):
                        buffer.seek(0)
                        dataframe = pd.read_csv(buffer, sep=delimiter, comment="#")
                        break

        if dataframe.empty:
            raise ValueError("Uploaded dataset is empty.")

        return dataframe

    # -------------------------------------------------------------------------
    def format_dataset_summary(self, dataframe: pd.DataFrame) -> str:
        """Produce a markdown overview of the dataset dimensions and missing values.

        Keyword arguments:
        dataframe -- Parsed dataset whose characteristics should be summarized.

        Return value:
        Multi-line markdown string describing dataset size and per-column missing
        value statistics.
        """
        rows, columns = dataframe.shape
        total_nans = int(dataframe.isna().sum().sum())
        column_lines: list[str] = []
        for name, series in dataframe.items():
            dtype = self.escape_markdown_table_cell(series.dtype)
            missing = int(series.isna().sum())
            safe_name = self.escape_markdown_table_cell(name)
            column_lines.append(f"| `{safe_name}` | `{dtype}` | {missing} |")

        summary_lines = [
            "### Dataset overview",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Rows | {rows} |",
            f"| Columns | {columns} |",
            f"| NaN cells | {total_nans} |",
            "",
            "### Column details",
            "",
            "| Column | Dtype | Missing |",
            "|---|---|---:|",
            *column_lines,
        ]
        return "\n".join(summary_lines)

    # -------------------------------------------------------------------------
    def load_from_database(self, dataset_name: str) -> tuple[dict[str, Any], str]:
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            raise ValueError("Dataset name is required.")
        dataset_name = dataset_name.strip()

        serializer = DataSerializer()
        dataframe = serializer.load_table("ADSORPTION_DATA")
        if dataframe.empty:
            raise ValueError("No datasets found in the database.")

        if "dataset_name" in dataframe.columns:
            filtered = dataframe[dataframe["dataset_name"] == dataset_name].copy()
        else:
            filtered = dataframe.copy()

        if filtered.empty:
            raise ValueError(f"Dataset '{dataset_name}' was not found.")

        cleaned = filtered.drop(columns=["dataset_name"], errors="ignore")
        serializable = cleaned.where(pd.notna(cleaned), None)
        dataset_payload: dict[str, Any] = {
            "dataset_name": dataset_name,
            "columns": list(serializable.columns),
            "records": serializable.to_dict(orient="records"),
            "row_count": int(serializable.shape[0]),
        }
        summary = self.format_dataset_summary(cleaned)
        return dataset_payload, summary

    # -------------------------------------------------------------------------
    def get_dataset_names(self) -> list[str]:
        serializer = DataSerializer()
        dataframe = serializer.load_table("ADSORPTION_DATA")
        if dataframe.empty or "dataset_name" not in dataframe.columns:
            return []
        names = dataframe["dataset_name"].dropna().unique().tolist()
        cleaned = [str(name).strip() for name in names if str(name).strip()]
        return sorted(cleaned)

    # -------------------------------------------------------------------------
    def save_to_database(self, payload: dict[str, Any]) -> None:
        """Persist uploaded dataset to ADSORPTION_DATA table.

        Keyword arguments:
        payload -- Dataset payload containing records, columns, and dataset_name.
        """
        records = payload.get("records", [])
        columns = payload.get("columns", [])
        dataset_name = payload.get("dataset_name", "uploaded_dataset")

        if not records:
            return

        df = pd.DataFrame.from_records(records, columns=columns)
        df["dataset_name"] = dataset_name
        serializer = DataSerializer()
        serializer.save_raw_dataset(df)
