import os
import pandas as pd
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Float, Integer, String, UniqueConstraint, create_engine
from sqlalchemy.dialects.sqlite import insert

from NISTADS.app.constants import DATA_PATH, INFERENCE_PATH
from NISTADS.app.logger import logger

Base = declarative_base()


###############################################################################
class SingleComponentAdsorption(Base):
    __tablename__ = 'SINGLE_COMPONENT_ADSORPTION'
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
        UniqueConstraint('filename', 'temperature', 'pressure', 
                         'adsorbent_name', 'adsorbate_name'),
    )


###############################################################################
class BinaryMixtureAdsorption(Base):
    __tablename__ = 'BINARY_MIXTURE_ADSORPTION'
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
        UniqueConstraint('filename', 'temperature', 'adsorbent_name', 
                         'compound_1', 'compound_2', 'compound_1_pressure',
                         'compound_2_pressure'),
    )
    
        
###############################################################################
class Adsorbate(Base):
    __tablename__ = 'ADSORBATES'
    InChIKey = Column(String, primary_key=True)
    name = Column(String)
    InChICode = Column(String)
    formula = Column(String)
    synonyms = Column(String)
    adsorbate_molecular_weight = Column(Float)
    adsorbate_molecular_formula = Column(String)
    adsorbate_SMILE = Column(String)
    __table_args__ = (
        UniqueConstraint('InChIKey'),
    )

    
###############################################################################
class Adsorbent(Base):
    __tablename__ = 'ADSORBENTS'
    name = Column(String)
    hashkey = Column(String, primary_key=True)
    formula = Column(String)
    synonyms = Column(String)
    External_Resources = Column(String)
    adsorbent_molecular_weight = Column(Float)
    adsorbent_molecular_formula = Column(String)
    adsorbent_SMILE = Column(String)
    __table_args__ = (
        UniqueConstraint('hashkey'),
    )

    
###############################################################################
class TrainData(Base):
    __tablename__ = 'TRAIN_DATA'
    temperature = Column(Float, primary_key=True)
    pressure = Column(Float, primary_key=True)
    adsorbed_amount = Column(Float)
    encoded_adsorbent = Column(Float, primary_key=True)
    adsorbate_molecular_weight = Column(Float)
    adsorbate_name = Column(String, primary_key=True)
    adsorbate_encoded_SMILE = Column(String)
    __table_args__ = (
        UniqueConstraint('temperature', 'pressure', 
                         'encoded_adsorbent', 'adsorbate_name'),
    )


###############################################################################
class ValidationData(Base):
    __tablename__ = 'VALIDATION_DATA'
    temperature = Column(Float, primary_key=True)
    pressure = Column(Float, primary_key=True)
    adsorbed_amount = Column(Float)
    encoded_adsorbent = Column(Float, primary_key=True)
    adsorbate_molecular_weight = Column(Float)
    adsorbate_name = Column(String, primary_key=True)
    adsorbate_encoded_SMILE = Column(String)
    __table_args__ = (
        UniqueConstraint('temperature', 'pressure', 
                         'encoded_adsorbent', 'adsorbate_name'),
    )

    
###############################################################################
class PredictedAdsorption(Base):
    __tablename__ = 'PREDICTED_ADSORPTION'
    experiment = Column(String, primary_key=True)
    temperature = Column(Float, primary_key=True)
    adsorbent_name = Column(String, primary_key=True)
    adsorbate_name = Column(String, primary_key=True)
    pressure = Column(Float)
    adsorbed_amount = Column(Float)
    predicted_adsorbed_amount = Column(Float)
    __table_args__ = (
        UniqueConstraint('experiment', 'temperature', 
                         'adsorbent_name', 'adsorbate_name'),
    )
    

###############################################################################
class CheckpointSummary(Base):
    __tablename__ = 'CHECKPOINTS_SUMMARY'
    checkpoint_name = Column(String, primary_key=True)
    sample_size = Column(Float)
    validation_size = Column(Float)
    seed = Column(Integer)
    precision_bits = Column(Integer)
    epochs = Column(Integer)
    additional_epochs = Column(Integer)
    batch_size = Column(Integer)
    split_seed = Column(Integer)
    image_augmentation = Column(String)
    image_height = Column(Integer)
    image_width = Column(Integer)
    image_channels = Column(Integer)
    jit_compile = Column(String)
    jit_backend = Column(String)
    device = Column(String)
    device_id = Column(String)
    number_of_processors = Column(Integer)
    use_tensorboard = Column(String)
    lr_scheduler_initial_lr = Column(Float)
    lr_scheduler_constant_steps = Column(Float)
    lr_scheduler_decay_steps = Column(Float)
    __table_args__ = (
        UniqueConstraint('checkpoint_name'),
    )
    

# [DATABASE]
###############################################################################
class AdsorptionDatabase:

    def __init__(self): 
        self.db_path = os.path.join(DATA_PATH, 'NISTADS_database.db')
        self.engine = create_engine(f'sqlite:///{self.db_path}', echo=False, future=True)
        self.Session = sessionmaker(bind=self.engine, future=True)
        self.insert_batch_size = 50000
              
    #--------------------------------------------------------------------------       
    def initialize_database(self):
        Base.metadata.create_all(self.engine)          
        inference_path = os.path.join(INFERENCE_PATH, 'inference_adsorption_data.csv')
        logger.debug(f'Updating database from {inference_path}')                    
        dataset = pd.read_csv(inference_path, sep=';', encoding='utf-8')        
        self.save_predictions_table(dataset)   

    #--------------------------------------------------------------------------
    def upsert_dataframe(self, df: pd.DataFrame, table_cls):
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
            records = df.to_dict(orient='records')
            for i in range(0, len(records), self.insert_batch_size):
                batch = records[i:i + self.insert_batch_size]
                stmt = insert(table).values(batch)
                # Columns to update on conflict
                update_cols = {c: getattr(stmt.excluded, c) for c in batch[0] if c not in unique_cols}
                stmt = stmt.on_conflict_do_update(
                    index_elements=unique_cols,
                    set_=update_cols
                )
                session.execute(stmt)
            session.commit()
        finally:
            session.close()
       
    #--------------------------------------------------------------------------
    def load_dataset_tables(self):
        with self.engine.connect() as conn:
            adsorption_data = pd.read_sql_table("SINGLE_COMPONENT_ADSORPTION", conn)
            guest_data = pd.read_sql_table("ADSORBATES", conn)
            host_data = pd.read_sql_table("ADSORBENTS", conn)

        return adsorption_data, guest_data, host_data

    #--------------------------------------------------------------------------
    def load_train_and_validation_tables(self):       
        with self.engine.connect() as conn:
            train_data = pd.read_sql_table("TRAIN_DATA", conn)
            validation_data = pd.read_sql_table("VALIDATION_DATA", conn)

        return train_data, validation_data

    #--------------------------------------------------------------------------
    def load_inference_data_table(self):         
        with self.engine.connect() as conn:
            data = pd.read_sql_table("PREDICTED_ADSORPTION", conn)
        return data

    #--------------------------------------------------------------------------
    def save_experiments_table(self, single_components: pd.DataFrame, binary_mixture: pd.DataFrame):
        self.upsert_dataframe(single_components, SingleComponentAdsorption)
        self.upsert_dataframe(binary_mixture, BinaryMixtureAdsorption)

    #--------------------------------------------------------------------------
    def save_materials_table(self, adsorbates : pd.DataFrame, adsorbents : pd.DataFrame,):    
        if adsorbates is not None:
            self.upsert_dataframe(adsorbates, Adsorbate)
        if adsorbents is not None:
            self.upsert_dataframe(adsorbents, Adsorbent)

    #--------------------------------------------------------------------------
    def save_train_and_validation_tables(self, train_data : pd.DataFrame, validation_data : pd.DataFrame):         
        with self.engine.begin() as conn:
            train_data.to_sql(
                "TRAIN_DATA", conn, if_exists='replace', index=False)
            validation_data.to_sql(
                "VALIDATION_DATA", conn, if_exists='replace', index=False)

    #--------------------------------------------------------------------------
    def save_predictions_table(self, data : pd.DataFrame):      
        with self.engine.begin() as conn:
            data.to_sql(
                "PREDICTED_ADSORPTION", conn, if_exists='replace', index=False)

    #--------------------------------------------------------------------------
    def save_checkpoints_summary_table(self, data : pd.DataFrame):         
        self.upsert_dataframe(data, CheckpointSummary)
    
    

    
