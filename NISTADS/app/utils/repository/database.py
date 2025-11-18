from __future__ import annotations

import os
from typing import Any

import pandas as pd
import sqlalchemy
from sqlalchemy import Column, Float, Integer, String, UniqueConstraint, create_engine
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import declarative_base, sessionmaker

from NISTADS.app.utils.constants import DATA_PATH, DATA_SOURCE_PATH
from NISTADS.app.utils.logger import logger
from NISTADS.app.utils.singleton import singleton

Base = declarative_base()


###############################################################################
class SingleComponentAdsorption(Base):
    __tablename__ = "SINGLE_COMPONENT_ADSORPTION"
    filename = Column(String, primary_key=True)
    temperature = Column(Float, primary_key=True)
    adsorptionUnits = Column(String)
    pressureUnits = Column(String)
    adsorbent_name = Column(String, primary_key=True)
    adsorbate_name = Column(String, primary_key=True)
    pressure = Column(Float, primary_key=True)
    adsorbed_amount = Column(Float)
    composition = Column(Float)
    __table_args__ = (
        UniqueConstraint(
            "filename", "temperature", "pressure", "adsorbent_name", "adsorbate_name"
        ),
    )


###############################################################################
class BinaryMixtureAdsorption(Base):
    __tablename__ = "BINARY_MIXTURE_ADSORPTION"
    filename = Column(String, primary_key=True)
    temperature = Column(Float, primary_key=True)
    adsorptionUnits = Column(String)
    pressureUnits = Column(String)
    adsorbent_name = Column(String, primary_key=True)
    compound_1 = Column(String, primary_key=True)
    compound_2 = Column(String, primary_key=True)
    compound_1_composition = Column(Float)
    compound_2_composition = Column(Float)
    compound_1_pressure = Column(Float, primary_key=True)
    compound_2_pressure = Column(Float, primary_key=True)
    compound_1_adsorption = Column(Float)
    compound_2_adsorption = Column(Float)
    __table_args__ = (
        UniqueConstraint(
            "filename",
            "temperature",
            "adsorbent_name",
            "compound_1",
            "compound_2",
            "compound_1_pressure",
            "compound_2_pressure",
        ),
    )


###############################################################################
class Adsorbate(Base):
    __tablename__ = "ADSORBATES"
    InChIKey = Column(String, primary_key=True)
    name = Column(String)
    InChICode = Column(String)
    formula = Column(String)
    adsorbate_molecular_weight = Column(Float)
    adsorbate_molecular_formula = Column(String)
    adsorbate_SMILE = Column(String)
    __table_args__ = (UniqueConstraint("InChIKey"),)


###############################################################################
class Adsorbent(Base):
    __tablename__ = "ADSORBENTS"
    name = Column(String)
    hashkey = Column(String, primary_key=True)
    formula = Column(String)
    adsorbent_molecular_weight = Column(Float)
    adsorbent_molecular_formula = Column(String)
    adsorbent_SMILE = Column(String)
    __table_args__ = (UniqueConstraint("hashkey"),)


###############################################################################
class TrainingData(Base):
    __tablename__ = "TRAINING_DATASET"
    filename = Column(String, primary_key=True)
    temperature = Column(Float)
    pressure = Column(String)
    adsorbed_amount = Column(String)
    encoded_adsorbent = Column(Float)
    adsorbate_molecular_weight = Column(Float)
    adsorbate_encoded_SMILE = Column(String)
    split = Column(String)
    __table_args__ = (UniqueConstraint("filename"),)


###############################################################################
class PredictedAdsorption(Base):
    __tablename__ = "PREDICTED_ADSORPTION"
    checkpoint = Column(String, primary_key=True)
    filename = Column(String, primary_key=True)
    temperature = Column(Float, primary_key=True)
    adsorbent_name = Column(String, primary_key=True)
    adsorbate_name = Column(String, primary_key=True)
    pressure = Column(Float, primary_key=True)
    adsorbed_amount = Column(Float)
    predicted_adsorbed_amount = Column(Float)
    __table_args__ = (
        UniqueConstraint(
            "checkpoint",
            "filename",
            "temperature",
            "adsorbent_name",
            "adsorbate_name",
            "pressure",
        ),
    )


###############################################################################
class CheckpointSummary(Base):
    __tablename__ = "CHECKPOINTS_SUMMARY"
    checkpoint = Column(String, primary_key=True)
    sample_size = Column(Float)
    validation_size = Column(Float)
    seed = Column(Integer)
    precision = Column(Integer)
    epochs = Column(Integer)
    batch_size = Column(Integer)
    split_seed = Column(Integer)
    jit_compile = Column(String)
    has_tensorboard_logs = Column(String)
    initial_LR = Column(Float)
    constant_steps_LR = Column(Float)
    decay_steps_LR = Column(Float)
    target_LR = Column(Float)
    max_measurements = Column(Integer)
    SMILE_size = Column(Integer)
    attention_heads = Column(Integer)
    n_encoders = Column(Integer)
    embedding_dimensions = Column(Integer)
    train_loss = Column(Float)
    val_loss = Column(Float)
    train_R_square = Column(Float)
    val_R_square = Column(Float)
    __table_args__ = (UniqueConstraint("checkpoint"),)


# [DATABASE]
###############################################################################
@singleton
class NISTADSDatabase:
    def __init__(self) -> None:
        self.db_path = os.path.join(DATA_PATH, "database.db")
        self.inference_path = os.path.join(
            DATA_SOURCE_PATH, "inference_adsorption_data.csv"
        )
        self.engine = create_engine(
            f"sqlite:///{self.db_path}", echo=False, future=True
        )
        self.Session = sessionmaker(bind=self.engine, future=True)
        self.insert_batch_size = 1000

    # -------------------------------------------------------------------------
    def initialize_database(self) -> None:
        if not os.path.exists(self.db_path):
            Base.metadata.create_all(self.engine)

    # -------------------------------------------------------------------------
    def get_table_class(self, table_name: str) -> Any:
        for cls in Base.__subclasses__():
            if hasattr(cls, "__tablename__") and cls.__tablename__ == table_name:
                return cls
        raise ValueError(f"No table class found for name {table_name}")

    # -------------------------------------------------------------------------
    def update_database_from_sources(self) -> pd.DataFrame | None:
        """Load the bundled inference CSV and persist it into the database.

        Keyword arguments:
            None.
        Return value:
            DataFrame representing the ingested inference dataset or None if
            loading fails.
        """
        dataset = pd.read_csv(self.inference_path, sep=";", encoding="utf-8")
        self.save_into_database(dataset, "PREDICTED_ADSORPTION")

        return dataset

    # -------------------------------------------------------------------------
    def upsert_dataframe(self, data: pd.DataFrame, table_cls: Any) -> None:
        """Insert or update rows in batches using the table unique constraint.

        Keyword arguments:
            data -- Dataset to store, containing columns matching the SQLAlchemy
                table definition.
            table_cls -- Declarative SQLAlchemy class that maps to the destination
                table.
        Return value:
            None.
        """
        table = table_cls.__table__
        session = self.Session()
        try:
            unique_cols = []
            for uc in table.constraints:
                if isinstance(uc, UniqueConstraint):
                    unique_cols = uc.columns.keys()
                    break
            if not unique_cols:
                raise ValueError(f"No unique constraint found for {table_cls.__name__}")

            # Batch insertions for speed
            records = data.to_dict(orient="records")
            for i in range(0, len(records), self.insert_batch_size):
                batch = records[i : i + self.insert_batch_size]
                stmt = insert(table).values(batch)
                # Columns to update on conflict
                update_cols = {
                    c: getattr(stmt.excluded, c)  # type: ignore
                    for c in batch[0]
                    if c not in unique_cols
                }
                stmt = stmt.on_conflict_do_update(
                    index_elements=unique_cols, set_=update_cols
                )
                session.execute(stmt)
                session.commit()
            session.commit()
        finally:
            session.close()

    # -------------------------------------------------------------------------
    def load_from_database(self, table_name: str) -> pd.DataFrame:
        with self.engine.connect() as conn:
            data = pd.read_sql_table(table_name, conn)

        return data

    # -------------------------------------------------------------------------
    def save_into_database(self, data: pd.DataFrame, table_name: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(sqlalchemy.text(f'DELETE FROM "{table_name}"'))
            data.to_sql(table_name, conn, if_exists="append", index=False)

    # -------------------------------------------------------------------------
    def upsert_into_database(self, data: pd.DataFrame, table_name: str) -> None:
        table_cls = self.get_table_class(table_name)
        self.upsert_dataframe(data, table_cls)

    # -------------------------------------------------------------------------
    def export_all_tables_as_csv(
        self, export_dir: str, chunksize: int | None = None
    ) -> None:
        """Export every database table to CSV files with optional chunking.

        Keyword arguments:
            export_dir -- Destination directory where CSV files will be written.
            chunksize -- Optional number of rows per chunk to limit memory usage
                during the export.
        Return value:
            None.
        """
        os.makedirs(export_dir, exist_ok=True)
        with self.engine.connect() as conn:
            for table in Base.metadata.sorted_tables:
                table_name = table.name
                csv_path = os.path.join(export_dir, f"{table_name}.csv")

                # Build a safe SELECT for arbitrary table names (quote with "")
                query = sqlalchemy.text(f'SELECT * FROM "{table_name}"')

                if chunksize:
                    first = True
                    for chunk in pd.read_sql(query, conn, chunksize=chunksize):
                        chunk.to_csv(
                            csv_path,
                            index=False,
                            header=first,
                            mode="w" if first else "a",
                            encoding="utf-8",
                            sep=",",
                        )
                        first = False
                    # If no chunks were returned, still write the header row
                    if first:
                        pd.DataFrame(columns=[c.name for c in table.columns]).to_csv(
                            csv_path, index=False, encoding="utf-8", sep=","
                        )
                else:
                    df = pd.read_sql(query, conn)
                    if df.empty:
                        pd.DataFrame(columns=[c.name for c in table.columns]).to_csv(
                            csv_path, index=False, encoding="utf-8", sep=","
                        )
                    else:
                        df.to_csv(csv_path, index=False, encoding="utf-8", sep=",")

        logger.info(f"All tables exported to CSV at {os.path.abspath(export_dir)}")

    # -------------------------------------------------------------------------
    def delete_all_data(self) -> None:
        with self.engine.begin() as conn:
            for table in reversed(Base.metadata.sorted_tables):
                conn.execute(table.delete())


# -----------------------------------------------------------------------------
database = NISTADSDatabase()
