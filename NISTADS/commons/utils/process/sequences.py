import os
import numpy as np
import pandas as pd
from keras.api.preprocessing import sequence
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

from tqdm import tqdm
tqdm.pandas()
      
from NISTADS.commons.constants import CONFIG, DATA_PATH
from NISTADS.commons.logger import logger



# [MERGE DATASETS]
###############################################################################
class PressureUptakeSeriesProcess:

    def __init__(self):

        self.P_TARGET_COL = 'pressure' 
        self.Q_TARGET_COL = 'adsorbed_amount'        
        self.max_points = CONFIG['dataset']['MAX_POINTS']  
        self.min_points = CONFIG['dataset']['MIN_POINTS']    

    #--------------------------------------------------------------------------
    def remove_leading_zeros(self, dataframe: pd.DataFrame):
        
        def _inner_function(row):
            pressure_series = row[self.P_TARGET_COL]
            uptake_series = row[self.Q_TARGET_COL]
            # Find the index of the first non-zero element or get the last index if all are zeros
            no_zero_index = next((i for i, x in enumerate(pressure_series) if x != 0), len(pressure_series) - 1)                
            # Determine how many leading zeros were removed
            zeros_removed = max(0, no_zero_index - 1)                
            processed_pressure_series = pressure_series[zeros_removed:]             
            processed_uptake_series = uptake_series[zeros_removed:]

            return pd.Series([processed_pressure_series, processed_uptake_series])

        dataframe[[self.P_TARGET_COL, self.Q_TARGET_COL]] = dataframe.apply(_inner_function, axis=1)
        
        return dataframe  


    #--------------------------------------------------------------------------  
    def sequence_padding(self, dataset : pd.DataFrame):
            
        pad_value = -1
        dataset['pressure_in_Pascal'] = sequence.pad_sequences(dataset['pressure_in_Pascal'], 
                                                 maxlen=self.max_points, 
                                                 value=pad_value, 
                                                 dtype='float32', 
                                                 padding='post').tolist()  

        dataset['uptake_in_mmol_g'] = sequence.pad_sequences(dataset['uptake_in_mmol_g'], 
                                                 maxlen=self.max_points, 
                                                 value=pad_value, 
                                                 dtype='float32', 
                                                 padding='post').tolist()         

        return dataset
    
    #--------------------------------------------------------------------------
    def select_by_sequence_size(self, dataset : pd.DataFrame):
        
        dataset = dataset[dataset[self.P_TARGET_COL].apply(lambda x: self.min_points <= len(x) <= self.max_points)]

        return dataset
       
    
    # normalize sequences using a RobustScaler: X = X - median(X)/IQR(X)
    # flatten and reshape array to make it compatible with the scaler
    #--------------------------------------------------------------------------  
    def normalize_sequences(self, train, test, column):        
        
        normalizer = MinMaxScaler(feature_range=(0,1))
        sequence_array = np.array([item for sublist in train[column] for item in sublist]).reshape(-1, 1)         
        normalizer.fit(sequence_array)
        train[column] = train[column].apply(lambda x: normalizer.transform(np.array(x).reshape(-1, 1)).flatten())
        test[column] = test[column].apply(lambda x: normalizer.transform(np.array(x).reshape(-1, 1)).flatten())

        return train, test, normalizer
    

    
# normalize parameters
###############################################################################  
def normalize_parameters(train_X, train_Y, test_X, test_Y):

    '''
    Normalize the input features and output labels for training and testing data.
    This method normalizes the input features and output labels to facilitate 
    better model training and evaluation.

    Keyword Arguments:
        train_X (DataFrame): DataFrame containing the features of the training data.
        train_Y (list): List containing the labels of the training data.
        test_X (DataFrame): DataFrame containing the features of the testing data.
        test_Y (list): List containing the labels of the testing data.

    Returns:
        Tuple: A tuple containing the normalized training features, normalized training labels,
                normalized testing features, and normalized testing labels.
    
    '''        
    # cast float type for both the labels and the continuous features columns 
    norm_columns = ['temperature', 'mol_weight', 'complexity', 'heavy_atoms']       
    train_X[norm_columns] = train_X[norm_columns].astype(float)        
    test_X[norm_columns] = test_X[norm_columns].astype(float)
    
    # normalize the numerical features (temperature and physicochemical properties)      
    self.param_normalizer = MinMaxScaler(feature_range=(0, 1))
    train_X[norm_columns] = self.param_normalizer.fit_transform(train_X[norm_columns])
    test_X[norm_columns] = self.param_normalizer.transform(test_X[norm_columns])        

    return train_X, train_Y, test_X, test_Y 



# [TOKENIZERS]
###############################################################################
class PretrainedTokenizers:

    def __init__(self): 

        self.tokenizer_strings = {'distilbert': 'distilbert/distilbert-base-uncased',
                                  'bert': 'bert-base-uncased',
                                  'roberta': 'roberta-base',
                                  'gpt2': 'gpt2',
                                  'xlm': 'xlm-mlm-enfr-1024'}
    
    #--------------------------------------------------------------------------
    def get_tokenizer(self, tokenizer_name):

        if tokenizer_name not in self.tokenizer_strings:
            tokenizer_string = tokenizer_string
            logger.warning(f'{tokenizer_string} is not among preselected models.')
        else:
            tokenizer_string = self.tokenizer_strings[tokenizer_name]                            
        
        logger.info(f'Loading {tokenizer_string} for text tokenization...')
        tokenizer_path = os.path.join(TOKENIZERS_PATH, tokenizer_name)
        os.makedirs(tokenizer_path, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_string, cache_dir=tokenizer_path)
        vocabulary_size = len(tokenizer.vocab)            

        return tokenizer, vocabulary_size 

    
# [TOKENIZER]
###############################################################################
class TokenWizard:
    
    def __init__(self, configuration):        
        
        tokenizer_name = configuration["dataset"]["TOKENIZER"] 
        self.max_report_size = configuration["dataset"]["MAX_REPORT_SIZE"] 
        selector = PretrainedTokenizers()
        self.tokenizer, self.vocabulary_size = selector.get_tokenizer(tokenizer_name)         
    
    #--------------------------------------------------------------------------
    def tokenize_text_corpus(self, train_data : pd.DataFrame, validation_data : pd.DataFrame):

        '''      
        Tokenizes text data using the specified tokenizer and updates the input DataFrames.

        Keyword Arguments:
            train_data (pd.DataFrame): DataFrame containing training data with a 'text' column.
            validation_data (pd.DataFrame): DataFrame containing validation data with a 'text' column.

        Returns:
            tuple: A tuple containing two elements:
                - train_data (pd.DataFrame): DataFrame with an additional 'tokens' column containing 
                  tokenized version of the 'text' column as lists of token ids.
                - validation_data (pd.DataFrame): DataFrame with an additional 'tokens' column containing 
                  tokenized version of the 'text' column as lists of token ids.        

        '''        
        self.train_text = train_data['text'].to_list()
        self.validation_text = validation_data['text'].to_list()
        
        # tokenize train and validation text using loaded tokenizer 
        train_tokens = self.tokenizer(self.train_text, padding=True, truncation=True,
                                      max_length=self.max_report_size, return_tensors='pt')
        validation_tokens = self.tokenizer(self.validation_text, padding=True, truncation=True, 
                                           max_length=self.max_report_size, return_tensors='pt')       
        
        # extract only token ids from the tokenizer output
        train_tokens = train_tokens['input_ids'].numpy().tolist() 
        val_tokens = validation_tokens['input_ids'].numpy().tolist()

        # add tokenizer sequences to the source dataframe, now containing the paths,
        # original text and tokenized text
        train_data['tokens'] = [' '.join(map(str, ids)) for ids in train_tokens]  
        validation_data['tokens'] = [' '.join(map(str, ids)) for ids in val_tokens]        
        
        return train_data, validation_data
    
  