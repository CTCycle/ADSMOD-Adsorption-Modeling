import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QImage, QPixmap

from NISTADS.commons.utils.data.serializer import DataSerializer, ModelSerializer
from NISTADS.commons.utils.data.loader import TrainingDataLoader, InferenceDataLoader
from NISTADS.commons.utils.data.builder import BuildAdsorptionDataset
from NISTADS.commons.utils.data.API import AdsorptionDataFetch, GuestHostDataFetch
from NISTADS.commons.utils.data.properties import MolecularProperties
from NISTADS.commons.utils.process.sequences import PressureUptakeSeriesProcess, SMILETokenization
from NISTADS.commons.utils.process.conversion import PQ_units_conversion
from NISTADS.commons.utils.learning.training.fitting import ModelTraining
from NISTADS.commons.utils.learning.models.qmodel import SCADSModel
from NISTADS.commons.utils.validation.checkpoints import ModelEvaluationSummary
from NISTADS.commons.utils.validation.dataset import AdsorptionPredictionsQuality
from NISTADS.commons.utils.learning.inference.predictor import AdsorptionPredictions
from NISTADS.commons.utils.process.sanitizer import (DataSanitizer, AggregateDatasets, 
                                                     TrainValidationSplit, FeatureNormalizer, 
                                                     AdsorbentEncoder) 

from NISTADS.commons.interface.workers import check_thread_status, update_progress_callback


from NISTADS.commons.logger import logger



###############################################################################
class GraphicsHandler:

    def __init__(self): 
        self.image_encoding = cv2.IMREAD_UNCHANGED
        self.gray_scale_encoding = cv2.IMREAD_GRAYSCALE
        self.BGRA_encoding = cv2.COLOR_BGRA2RGBA
        self.BGR_encoding = cv2.COLOR_BGR2RGB

    #--------------------------------------------------------------------------
    def convert_fig_to_qpixmap(self, fig):    
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        # get the size in pixels and initialize raw RGBA buffer
        width, height = canvas.get_width_height()        
        buf = canvas.buffer_rgba()
        # construct a QImage pointing at that memory (no PNG decoding)
        qimg = QImage(buf, width, height, QImage.Format_RGBA8888)

        return QPixmap.fromImage(qimg)
    
    #--------------------------------------------------------------------------    
    def load_image_as_pixmap(self, path):    
        img = cv2.imread(path, self.image_encoding)
        # Handle grayscale, RGB, or RGBA
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, self.gray_scale_encoding)
        elif img.shape[2] == 4:  # BGRA
            img = cv2.cvtColor(img, self.BGRA_encoding)
        else:  # BGR
            img = cv2.cvtColor(img, self.BGR_encoding)

        h, w = img.shape[:2]
        if img.shape[2] == 3:
            qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
        else:  
            qimg = QImage(img.data, w, h, 4 * w, QImage.Format_RGBA8888)

        return QPixmap.fromImage(qimg)
    

###############################################################################
class DatasetEvents:

    def __init__(self, database, configuration): 
        self.database = database                  
        self.configuration = configuration 

    #--------------------------------------------------------------------------
    def run_data_collection_pipeline(self, progress_callback=None, worker=None):      
        # get isotherm indexes invoking API
        logger.info('Collect adsorption isotherm indices from NIST-ARPA-E database')
        API = AdsorptionDataFetch(self.configuration)
        experiments_index = API.get_experiments_index() 

        logger.info('Extracting adsorption isotherms data from JSON response')
        adsorption_data = API.get_experiments_data(
            experiments_index, worker=worker, progress_callback=progress_callback)
        
        # remove excluded columns from the dataframe
        builder = BuildAdsorptionDataset()
        logger.info('Cleaning and processing adsorption dataset')
        adsorption_data = builder.drop_excluded_columns(adsorption_data)
        # split current dataframe by complexity of the mixture (single component or binary mixture)
        logger.info('Experiments will be split based on mixture complexity')
        single_component, binary_mixture = builder.split_by_mixture_complexity(adsorption_data) 
        # extract nested data in dataframe rows and reorganise them into columns
        single_component = builder.extract_nested_data(single_component)
        binary_mixture = builder.extract_nested_data(binary_mixture)

        # expand the dataset to represent each measurement with a single row
        # save the final version of the adsorption dataset  
        serializer = DataSerializer(self.database, self.configuration)
        single_component, binary_mixture = builder.expand_dataset(
            single_component, binary_mixture)
        serializer.save_adsorption_datasets(single_component, binary_mixture)     
        logger.info('Experiments data collection is concluded')  

        # get guest and host indexes invoking API
        logger.info('Collect guest and host indices from NIST-ARPA-E database')
        API = GuestHostDataFetch(self.configuration)
        guest_index, host_index = API.get_materials_index()    
        # extract adsorbents and sorbates data from relative indices        
        logger.info('Extracting adsorbents and sorbates data from relative indices')
        guest_data, host_data = API.get_materials_data(
            guest_index, host_index, worker=worker, progress_callback=progress_callback)         
        
        # save the final version of the materials dataset 
        serializer.save_materials_datasets(guest_data, host_data)
        logger.info('Materials data collection is concluded')

    #--------------------------------------------------------------------------
    def run_chemical_properties_pipeline(self, progress_callback=None, worker=None):         
        serializer = DataSerializer(self.database, self.configuration)
        experiments, guest_data, host_data = serializer.load_source_datasets()         
           
        properties = MolecularProperties(self.configuration)  
        # process guest (adsorbed species) data by adding molecular properties
        logger.info('Retrieving molecular properties for sorbate species using PubChem API')
        guest_data = properties.fetch_guest_properties(
            experiments, guest_data, worker=worker, progress_callback=progress_callback) 
        # save the final version of the materials dataset    
        serializer.save_materials_datasets(
            guest_data=guest_data)

        # process host (adsorbent materials) data by adding molecular properties   
        logger.info('Retrieving molecular properties for adsorbent materials using PubChem API') 
        host_data = properties.fetch_host_properties(
            experiments, host_data, worker=worker, progress_callback=progress_callback) 
        # save the final version of the materials dataset    
        serializer.save_materials_datasets(host_data=host_data)      
    
    #--------------------------------------------------------------------------
    def run_dataset_builder(self, progress_callback=None, worker=None):        
        serializer = DataSerializer(self.database, self.configuration)
        adsorption_data, guest_data, host_data = serializer.load_source_datasets() 
        logger.info(f'{adsorption_data.shape[0]} measurements in the dataset')
        logger.info(f'{guest_data.shape[0]} total guests (adsorbates species) in the dataset')
        logger.info(f'{host_data.shape[0]} total hosts (adsorbent materials) in the dataset')
        
        # group single component data based on the experiment name 
        # merge adsorption data with retrieved materials properties (guest and host)
        aggregator = AggregateDatasets(self.configuration)
        processed_data = aggregator.aggregate_adsorption_measurements(adsorption_data)

        # check thread for interruption 
        check_thread_status(worker)
        update_progress_callback(1, 7, progress_callback)
        # start joining materials properties
        processed_data = aggregator.join_materials_properties(processed_data, guest_data, host_data)
        logger.info(f'Dataset has been aggregated for a total of {processed_data.shape[0]} experiments')         

        # check thread for interruption 
        check_thread_status(worker)
        update_progress_callback(2, 7, progress_callback)
        # convert pressure and uptake into standard units:
        # pressure to Pascal, uptake to mol/g
        sequencer = PressureUptakeSeriesProcess(self.configuration)
        logger.info('Converting pressure into Pascal and uptake into mol/g')   
        processed_data = PQ_units_conversion(processed_data) 

        # check thread for interruption 
        check_thread_status(worker)
        update_progress_callback(3, 7, progress_callback)
        # exlude all data outside given boundaries, such as negative temperature values 
        # and pressure and uptake values below zero or above upper limits
        sanitizer = DataSanitizer(self.configuration)  
        processed_data = sanitizer.exclude_OOB_values(processed_data)
        
        # remove repeated zero values at the beginning of pressure and uptake series  
        # then filter out experiments with not enough measurements 
        processed_data = sequencer.remove_leading_zeros(processed_data)   
        processed_data = sequencer.filter_by_sequence_size(processed_data)          

        # check thread for interruption 
        check_thread_status(worker)
        update_progress_callback(4, 7, progress_callback)
        # perform SMILE sequence tokenization  
        tokenization = SMILETokenization(self.configuration) 
        logger.info('Tokenizing SMILE sequences for adsorbate species')   
        processed_data, smile_vocab = tokenization.process_SMILE_sequences(processed_data)  

        # split data into train set and validation set
        logger.info('Preparing dataset of images and captions based on splitting size')  
        splitter = TrainValidationSplit(self.configuration, processed_data)     
        train_data, validation_data = splitter.split_train_and_validation() 

        # check thread for interruption 
        check_thread_status(worker)
        update_progress_callback(5, 7, progress_callback)
        # normalize pressure and uptake series using max values computed from 
        # the training set, then pad sequences to a fixed length
        normalizer = FeatureNormalizer(self.configuration, train_data)
        train_data = normalizer.normalize_molecular_features(train_data) 
        train_data = normalizer.PQ_series_normalization(train_data) 
        validation_data = normalizer.normalize_molecular_features(validation_data) 
        validation_data = normalizer.PQ_series_normalization(validation_data)      
    
        # check thread for interruption 
        check_thread_status(worker)
        update_progress_callback(6, 7, progress_callback)
        # add padding to pressure and uptake series to match max length
        train_data = sequencer.PQ_series_padding(train_data)     
        validation_data = sequencer.PQ_series_padding(validation_data)     

        
        encoding = AdsorbentEncoder(self.configuration, train_data)    
        train_data = encoding.encode_adsorbents_by_name(train_data)
        validation_data = encoding.encode_adsorbents_by_name(validation_data)    
        adsorbent_vocab = encoding.mapping

        # check thread for interruption 
        check_thread_status(worker)
        update_progress_callback(7, 7, progress_callback)
      
        # save preprocessed data using data serializer   
        train_data = sanitizer.isolate_processed_features(train_data)
        validation_data = sanitizer.isolate_processed_features(validation_data)           
        serializer.save_train_and_validation_data(
            train_data, validation_data, smile_vocab, 
            adsorbent_vocab, normalizer.statistics) 

    # define the logic to handle successfull data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_success(self, window, message):
        # send message to status bar
        window.statusBar().showMessage(message)
    
    # define the logic to handle error during data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_error(self, window, err_tb):
        exc, tb = err_tb
        logger.error(f"{exc}\n{tb}")
        QMessageBox.critical(window, 'Something went wrong!', f"{exc}\n\n{tb}")             



###############################################################################
class ValidationEvents:

    def __init__(self, database, configuration):
        self.database = database       
        self.configuration = configuration 
        
    #--------------------------------------------------------------------------
    def run_dataset_evaluation_pipeline(self, metrics, progress_callback=None, worker=None):
        sample_size = self.configuration.get("sample_size", 1.0)
        serializer = DataSerializer(self.configuration)
        adsorption_data, guest_data, host_data = serializer.load_source_datasets() 
        logger.info(f'{adsorption_data.shape[0]} measurements in the dataset')
        logger.info(f'{guest_data.shape[0]} adsorbates species in the dataset')
        logger.info(f'{host_data.shape[0]} adsorbent materials in the dataset')

        # load preprocessed data and associated metadata        
        train_data, val_data, metadata, vocabularies = serializer.load_train_and_validation_data()

        # check thread for interruption 
        check_thread_status(worker)

        # add adsorption data analysis   

        images = []  
        if 'experiments_clustering' in metrics:
            logger.info('Current metric: Adsorption isotherm clustering')
            # images.append(self.analyzer.calculate_pixel_intensity_distribution(
            #     images_paths, progress_callback, worker))

        return images 

    #--------------------------------------------------------------------------
    def get_checkpoints_summary(self, progress_callback=None, worker=None): 
        summarizer = ModelEvaluationSummary(self.configuration)    
        checkpoints_summary = summarizer.get_checkpoints_summary(progress_callback, worker) 
        logger.info(f'Checkpoints summary has been created for {checkpoints_summary.shape[0]} models')    
    
    #--------------------------------------------------------------------------
    def run_model_evaluation_pipeline(self, metrics, selected_checkpoint, device='CPU', 
                                      progress_callback=None, worker=None):
        modser = ModelSerializer()         
        model, train_config, session, checkpoint_path = modser.load_checkpoint(
            selected_checkpoint)    
        model.summary(expand_nested=True)

        # load preprocessed data and associated metadata
        serializer = DataSerializer(train_config)
        _, val_data, metadata, vocabularies = serializer.load_train_and_validation_data()
        logger.info(f'Validation data has been loaded: {val_data.shape[0]} samples')    
    
        loader = InferenceDataLoader(train_config)      
        validation_dataset = loader.build_inference_dataloader(val_data)

        # compare reconstructed isotherms from predictions    
        validator = AdsorptionPredictionsQuality(
            model, train_config, metadata, checkpoint_path)      
        validator.visualize_adsorption_isotherms(val_data)       
            
        images = []
        if 'evaluation_report' in metrics:
            # evaluate model performance over the training and validation dataset 
            summarizer = ModelEvaluationSummary(self.database, self.configuration)       
            summarizer.get_evaluation_report(model, validation_dataset, worker=worker)

        if 'adsorption_isotherms_prediction' in metrics:
            validator = AdsorptionPredictionsQuality(
            model, train_config, metadata, checkpoint_path)      
            images.append(validator.visualize_adsorption_isotherms(
                val_data, progress_callback, worker=worker))                    

        return images      

    # define the logic to handle successfull data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_success(self, window, message):
        # send message to status bar
        window.statusBar().showMessage(message)
    
    # define the logic to handle error during data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_error(self, window, err_tb):
        exc, tb = err_tb
        logger.error(f"{exc}\n{tb}")
        QMessageBox.critical(window, 'Something went wrong!', f"{exc}\n\n{tb}")  

   

###############################################################################
class ModelEvents:

    def __init__(self, database, configuration): 
        self.database = database        
        self.configuration = configuration 

    #--------------------------------------------------------------------------
    def get_available_checkpoints(self):
        serializer = ModelSerializer()
        return serializer.scan_checkpoints_folder()
            
    #--------------------------------------------------------------------------
    def run_training_pipeline(self, progress_callback=None, worker=None):  
        dataserializer = DataSerializer(self.database, self.configuration)        
        train_data, val_data, metadata, vocabularies = dataserializer.load_train_and_validation_data() 

        logger.info('Building model data loaders with prefetching and parallel processing')   
        builder = TrainingDataLoader(self.configuration)   
        train_dataset, validation_dataset = builder.build_training_dataloader(
            train_data, val_data) 
        
        # check thread for interruption 
        check_thread_status(worker)
        
        modser = ModelSerializer()
        checkpoint_path = modser.create_checkpoint_folder()        
         
        # set device for training operations based on user configuration        
        logger.info('Setting device for training operations based on user configuration') 
        trainer = ModelTraining(self.configuration)           
        trainer.set_device() 
       
        # Setting callbacks and training routine for the machine learning model           
        logger.info('Building SCADS model based on user configuration')  
        wrapper = SCADSModel(metadata, self.configuration)
        model = wrapper.get_model(model_summary=True) 

        # generate graphviz plot fo the model layout       
        modser.save_model_plot(model, checkpoint_path)   

        # check thread for interruption 
        check_thread_status(worker)           

        # perform training and save model at the end
        trainer.train_model(model, train_dataset, validation_dataset, checkpoint_path,
                            progress_callback=progress_callback, worker=worker)
        
    #--------------------------------------------------------------------------
    def resume_training_pipeline(self, selected_checkpoint, progress_callback=None, 
                                 worker=None):        
        
        logger.info(f'Loading {selected_checkpoint} checkpoint from pretrained models')   
        modser = ModelSerializer()      
        model, train_config, session, checkpoint_path = modser.load_checkpoint(
            selected_checkpoint)    
        model.summary(expand_nested=True)  

        # setting device for training    
        trainer = ModelTraining(self.configuration, metadata)    
        trainer.set_device()

        # check thread for interruption 
        check_thread_status(worker)     
          
        logger.info('Loading preprocessed data and building dataloaders')     
        dataserializer = DataSerializer(self.database, train_config)
        train_data, val_data, metadata, vocabularies = dataserializer.load_train_and_validation_data()
           
        builder = TrainingDataLoader(train_config)   
        train_dataset, validation_dataset = builder.build_training_dataloader(
            train_data, val_data)
        
        # check worker status to allow interruption
        check_thread_status(worker)    
        
        # Setting callbacks and training routine for the machine learning model        
        # resume training from pretrained model    
        trainer.train_model(
            model, train_dataset, validation_dataset, checkpoint_path, session,
            progress_callback=progress_callback, worker=worker)    
        
    #--------------------------------------------------------------------------
    def run_inference_pipeline(self, selected_checkpoint, device='CPU', 
                               progress_callback=None, worker=None):
        logger.info(f'Loading {selected_checkpoint} checkpoint from pretrained models')
        modser = ModelSerializer()         
        model, train_config, session, checkpoint_path = modser.load_checkpoint(
            selected_checkpoint)    
        model.summary(expand_nested=True)  
        
        # setting device for training         
        trainer = ModelTraining(train_config)    
        trainer.set_device()

        # check worker status to allow interruption
        check_thread_status(worker)  

        # select images from the inference folder and retrieve current paths     
        serializer = DataSerializer(self.database, self.configuration)     
        inference_data = serializer.load_inference_data()  

        logger.info('Preprocessing inference input data according to model configuration')
        predictor = AdsorptionPredictions(model, train_config, checkpoint_path)
        predictions = predictor.predict_adsorption_isotherm(
            inference_data, progress_callback=progress_callback, worker=worker)
        
        predictions_dataset = predictor.merge_predictions_to_dataset(inference_data, predictions)
        serializer.save_predictions_dataset(predictions_dataset)
        logger.info('Predictions dataset saved successfully in database') 

        
    # define the logic to handle successfull data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_success(self, window, message):
        # send message to status bar
        window.statusBar().showMessage(message)
    
    # define the logic to handle error during data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_error(self, window, err_tb):
        exc, tb = err_tb
        logger.error(f"{exc}\n{tb}")
        QMessageBox.critical(window, 'Something went wrong!', f"{exc}\n\n{tb}")  

