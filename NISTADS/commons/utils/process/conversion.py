import pandas as pd
from tqdm import tqdm

from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger


###############################################################################
def PQ_units_conversion(dataframe : pd.DataFrame):
    P_converter = PressureConversion()
    Q_converter = UptakeConversion()
    converted_data = P_converter.convert_pressure_units(dataframe)
    converted_data = Q_converter.convert_uptake_data(converted_data)

    return converted_data

      

# [CONVERSION OF PRESSURE]
###############################################################################
class PressureConversion:

    def __init__(self):        
        self.P_COL = 'pressure'
        self.P_UNIT_COL = 'pressureUnits'     
        self.conversions = {'bar' : self.bar_to_Pascal}                                        
       
    #--------------------------------------------------------------------------
    def bar_to_Pascal(self, p_vals):                    
        return [int(p_val * 100000) for p_val in p_vals]    

    #--------------------------------------------------------------------------
    def convert_pressure_units(self, dataframe: pd.DataFrame):        
        dataframe[self.P_COL] = dataframe.apply(
            lambda row: self.conversions.get(row[self.P_UNIT_COL], lambda x: x)(row[self.P_COL]), axis=1)           
        dataframe.drop(columns=self.P_UNIT_COL, inplace=True)    

        return dataframe
    


# [CONVERSION OF UPTAKE]
###############################################################################
class UptakeConversion:  

    def __init__(self):        
        self.Q_COL = 'adsorbed_amount'
        self.Q_UNIT_COL = 'adsorptionUnits'          
        self.mol_W = 'adsorbate_molecular_weight'

        # Dictionary mapping units to their respective conversion methods
        self.conversions = {'mmol/g': self.convert_mmol_g_or_mol_kg,
                            'mol/kg': self.convert_mmol_g_or_mol_kg,
                            'mmol/kg': self.convert_mmol_kg,
                            'mg/g': self.convert_mg_g,
                            'g/g': self.convert_g_g,
                            'wt%': self.convert_wt_percent,
                            'g Adsorbate / 100g Adsorbent': self.convert_g_adsorbate_per_100g_adsorbent,
                            'g/100g': self.convert_g_adsorbate_per_100g_adsorbent,
                            'ml(STP)/g': self.convert_ml_stp_g_or_cm3_stp_g,
                            'cm3(STP)/g': self.convert_ml_stp_g_or_cm3_stp_g}

    #--------------------------------------------------------------------------
    def convert_mmol_g_or_mol_kg(self, q_vals):
        return [q_val for q_val in q_vals]

    #--------------------------------------------------------------------------
    def convert_mmol_kg(self, q_vals):
        return [q_val / 1000 for q_val in q_vals]

    #--------------------------------------------------------------------------
    def convert_mg_g(self, q_vals, mol_weight):
        return [q_val / float(mol_weight) for q_val in q_vals]

    #--------------------------------------------------------------------------
    def convert_g_g(self, q_vals, mol_weight):
        return [q_val / float(mol_weight) * 1000 for q_val in q_vals]

    #--------------------------------------------------------------------------
    def convert_wt_percent(self, q_vals, mol_weight):
        return [(q_val / 100) / float(mol_weight) * 1000 for q_val in q_vals]

    #--------------------------------------------------------------------------
    def convert_g_adsorbate_per_100g_adsorbent(self, q_vals, mol_weight):
        return [(q_val / 100) / float(mol_weight) * 1000 for q_val in q_vals]

    #--------------------------------------------------------------------------
    def convert_ml_stp_g_or_cm3_stp_g(self, q_vals):
        return [q_val / 22.414 * 1000 for q_val in q_vals]

    #--------------------------------------------------------------------------
    def convert_uptake_data(self, dataframe: pd.DataFrame):        
        dataframe[self.Q_COL] = dataframe.apply(
            lambda row: (self.conversions.get(row[self.Q_UNIT_COL], lambda x, *args: x)(
                row[self.Q_COL], *(row[self.mol_W],) if row[self.Q_UNIT_COL] in {
                    'mg/g', 'g/g', 'wt%', 'g Adsorbate / 100g Adsorbent', 'g/100g'} else ())), axis=1)
               
        dataframe.drop(columns=self.Q_UNIT_COL, inplace=True)    
           
        return dataframe

        
    
    

   
  


    

        
    
    

    
 

    