import re
import numpy as np
import pandas as pd
from keras.api.preprocessing import sequence
from tqdm import tqdm
      
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger



# [MERGE DATASETS]
###############################################################################
class PressureUptakeSeriesProcess:

    def __init__(self, configuration):
        self.P_COL = 'pressure' 
        self.Q_COL = 'adsorbed_amount'        
        self.max_points = configuration['dataset']['MAX_PQ_POINTS']  
        self.min_points = configuration['dataset']['MIN_PQ_POINTS'] 
        self.max_pressure = configuration['dataset']['MAX_PRESSURE']
        self.max_uptake = configuration['dataset']['MAX_UPTAKE']     
        self.pad_value = -1

    #--------------------------------------------------------------------------
    def remove_leading_zeros(self, dataframe: pd.DataFrame):
        
        def _inner_function(row):
            pressure_series = row[self.P_COL]
            uptake_series = row[self.Q_COL]
            # Find the index of the first non-zero element or get the last index if all are zeros
            no_zero_index = next((i for i, x in enumerate(pressure_series) if x != 0), len(pressure_series) - 1)                
            # Determine how many leading zeros were removed
            zeros_removed = max(0, no_zero_index - 1)                
            processed_pressure_series = pressure_series[zeros_removed:]             
            processed_uptake_series = uptake_series[zeros_removed:]

            return pd.Series([processed_pressure_series, processed_uptake_series])

        dataframe[[self.P_COL, self.Q_COL]] = dataframe.apply(_inner_function, axis=1)
        
        return dataframe
    
    #--------------------------------------------------------------------------  
    def PQ_series_padding(self, dataset : pd.DataFrame):            
        
        dataset[self.P_COL] = sequence.pad_sequences(
            dataset[self.P_COL], maxlen=self.max_points, value=self.pad_value, 
            dtype='float32', padding='post').tolist()  

        dataset[self.Q_COL] = sequence.pad_sequences(
            dataset[self.Q_COL], maxlen=self.max_points, value=self.pad_value, 
            dtype='float32', padding='post').tolist()          

        return dataset
    
    #--------------------------------------------------------------------------  
    def series_normalization(self, dataset : pd.DataFrame):
        dataset[self.P_COL] = dataset[self.P_COL].apply(
            lambda x : [v/self.max_pressure if v != self.pad_value else v for v in x])
                    
        dataset[self.Q_COL] = dataset[self.Q_COL].apply(
            lambda x : [v/self.max_uptake if v != self.pad_value else v for v in x])

        return dataset
    
    #--------------------------------------------------------------------------
    def filter_by_sequence_size(self, dataset : pd.DataFrame):        
        dataset = dataset[dataset[self.P_COL].apply(
            lambda x: self.min_points <= len(x) <= self.max_points)]

        return dataset
        
    #--------------------------------------------------------------------------
    def convert_to_values_string(self, dataset : pd.DataFrame):        
        dataset[self.P_COL] = dataset[self.P_COL].apply(lambda x : ' '.join(map(str, x))) 
        dataset[self.Q_COL] = dataset[self.Q_COL].apply(lambda x : ' '.join(map(str, x)))

        return dataset
       

  

# [TOKENIZERS]
###############################################################################
class SMILETokenization:

    def __init__(self, configuration): 

        self.element_symbols = [
            'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P',
            'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
            'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc',
            'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La',
            'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
            'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At',
            'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es',
            'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Fl',
            'Lv', 'Ts', 'Og']
        
        self.organic_subset = ['B', 'C', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I']
        self.SMILE_padding = configuration['dataset']['SMILE_PADDING']
        self.pad_value = -1
        
    #--------------------------------------------------------------------------
    def tokenize_SMILE_string(self, SMILE):     

        tokens = []
        if isinstance(SMILE, str):  
            tokens = []          
            i = 0
            length = len(SMILE)
            while i < length:     
                c = SMILE[i]
                # Handle brackets
                if c == '[':
                    # Start of bracketed atom
                    j = i + 1
                    bracket_content = ''
                    while j < length and SMILE[j] != ']':
                        bracket_content += SMILE[j]
                        j += 1
                    if j == length:
                        logger.error(f'Incorrect SMILE sequence: {SMILE}')
                    # Now bracket_content contains the content inside brackets
                    # We need to extract the element symbol from bracket_content
                    # The element symbol is the first one or two letters after optional isotope
                    m = re.match(r'(\d+)?([A-Z][a-z]?)', bracket_content)
                    if not m:
                        raise ValueError(f"Invalid atom in brackets: [{bracket_content}]")
                    isotope, element = m.groups()
                    if element not in self.element_symbols:
                        raise ValueError(f"Unknown element symbol: {element}")
                    tokens.append('[' + bracket_content + ']')
                    i = j + 1  # Move past the closing ']'
                # Handle ring closures with '%'
                elif c == '%':
                    # Ring closure with numbers greater than 9
                    if i + 2 < length and SMILE[i+1:i+3].isdigit():
                        tokens.append(SMILE[i:i+3])
                        i += 3
                    else:
                        logger.error(f"Invalid ring closure with '%' in SMILE string {SMILE}")
                # Handle two-character organic subset elements outside brackets
                elif c == 'C' and i + 1 < length and SMILE[i+1] == 'l':
                    tokens.append('Cl')
                    i += 2
                elif c == 'B' and i + 1 < length and SMILE[i+1] == 'r':
                    tokens.append('Br')
                    i += 2
                # Handle one-character organic subset elements outside brackets
                elif c in 'BCNOPSFHI':
                    tokens.append(c)
                    i += 1
                # Handle aromatic atoms (lowercase letters)
                elif c in 'bcnops':
                    tokens.append(c)
                    i +=1
                # Handle digits for ring closures
                elif c.isdigit():
                    tokens.append(c)
                    i +=1
                # Handle bond symbols
                elif c in '-=#:/$\\':
                    tokens.append(c)
                    i +=1
                # Handle branch symbols
                elif c in '()':
                    tokens.append(c)
                    i +=1
                # Handle chirality '@'
                elif c == '@':
                    # Check if the next character is also '@'
                    if i +1 < length and SMILE[i+1] == '@':
                        tokens.append('@@')
                        i +=2
                    else:
                        tokens.append('@')
                        i +=1
                # Handle '+' or '-' charges
                elif c == '+' or c == '-':
                    charge = c
                    j = i +1
                    while j < length and (SMILE[j] == c or SMILE[j].isdigit()):
                        charge += SMILE[j]
                        j +=1
                    tokens.append(charge)
                    i = j
                # Handle wildcard '*'
                elif c == '*':
                    tokens.append(c)
                    i +=1
                else:
                    logger.debug(f"Unrecognized character '{c}' at position {i}")
                    i += 1

        return tokens
    
    #--------------------------------------------------------------------------
    def SMILE_tokens_encoding(self, data: pd.DataFrame): 

        adsorbate_SMILE_tokens = set(token for tokens in data['adsorbate_tokenized_SMILE'] for token in tokens)   
        adsorbent_SMILE_tokens = set(token for tokens in data['adsorbent_tokenized_SMILE'] for token in tokens)  
        all_SMILE_tokens = adsorbate_SMILE_tokens | adsorbent_SMILE_tokens        
             
        # Map each token to a unique integer
        token_to_id = {token: idx for idx, token in enumerate(sorted(all_SMILE_tokens))}
        id_to_token = {idx: token for token, idx in token_to_id.items()}        
        # Apply the encoding to each tokenized SMILE
        data['adsorbate_encoded_SMILE'] = data['adsorbate_tokenized_SMILE'].apply(
            lambda tokens: [int(token_to_id[token]) for token in tokens])
        
        data['adsorbent_encoded_SMILE'] = data['adsorbent_tokenized_SMILE'].apply(
            lambda tokens: [int(token_to_id[token]) for token in tokens])
        
        return data, id_to_token
    
    #--------------------------------------------------------------------------  
    def SMILE_series_padding(self, dataset : pd.DataFrame):       
        dataset['adsorbate_encoded_SMILE'] = sequence.pad_sequences(
            dataset['adsorbate_encoded_SMILE'], maxlen=self.SMILE_padding, 
            value=self.pad_value, dtype='float32', padding='post').tolist() 

        dataset['adsorbent_encoded_SMILE'] = sequence.pad_sequences(
            dataset['adsorbent_encoded_SMILE'], maxlen=self.SMILE_padding, 
            value=self.pad_value, dtype='float32', padding='post').tolist()         

        return dataset
    
    #--------------------------------------------------------------------------
    def process_SMILE_data(self, data : pd.DataFrame):
        data['adsorbate_tokenized_SMILE'] = data['adsorbate_SMILE'].apply(
            lambda x : self.tokenize_SMILE_string(x)) 
        data['adsorbent_tokenized_SMILE'] = data['adsorbent_SMILE'].apply(
            lambda x : self.tokenize_SMILE_string(x))

        data, smile_vocabulary = self.SMILE_tokens_encoding(data)        
        data = self.SMILE_series_padding(data)

        data['adsorbate_tokenized_SMILE'] = data['adsorbate_SMILE'].apply(
            lambda x: ' '.join(map(str, x)) if not isinstance(x, float) else x)
        data['adsorbent_tokenized_SMILE'] = data['adsorbent_SMILE'].apply(
            lambda x: ' '.join(map(str, x)) if not isinstance(x, float) else x)

        return data, smile_vocabulary


        
    