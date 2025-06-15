import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PySide6.QtWidgets import QMessageBox
from PySide6.QtGui import QImage, QPixmap

from NISTADS.commons.utils.data.serializer import DataSerializer, ModelSerializer
from NISTADS.commons.utils.data.loader import TrainingDataLoader, InferenceDataLoader
from NISTADS.commons.utils.data.builder import BuildAdsorptionDataset
from NISTADS.commons.utils.data.API import AdsorptionDataFetch, GuestHostDataFetch
from NISTADS.commons.utils.data.properties import MolecularProperties
from NISTADS.commons.interface.workers import check_thread_status

from NISTADS.commons.constants import *
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

    def __init__(self, configuration):                   
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
        serializer = DataSerializer(self.configuration)
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
        serializer = DataSerializer(self.configuration)
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
    def get_generated_report(self, image_name):               
        dataset = self.serializer.load_source_dataset(sample_size=1.0)
        image_no_ext = image_name.split('.')[0]  
        mask = dataset['image'].astype(str).str.contains(image_no_ext, case=False, na=False)
        description = dataset.loc[mask, 'text'].values
        description = description[0] if len(description) > 0 else self.text_placeholder  
        
        return description   
    
    #--------------------------------------------------------------------------
    def build_ML_dataset(self, progress_callback=None, worker=None):   
        sample_size = self.configuration.get("sample_size", 1.0)            
        dataset = self.serializer.load_source_dataset(sample_size=sample_size)
        
        # sanitize text corpus by removing undesired symbols and punctuation     
        sanitizer = TextSanitizer(self.configuration)
        processed_data = sanitizer.sanitize_text(dataset)
        logger.info(f'Dataset includes {processed_data.shape[0]} samples')

        # preprocess text corpus using selected pretrained tokenizer. Text is tokenized
        # into subunits and these are eventually mapped to integer indexes        
        tokenization = TokenWizard(self.configuration) 
        logger.info(f'Tokenizing text corpus using pretrained {tokenization.tokenizer_name} tokenizer')    
        processed_data = tokenization.tokenize_text_corpus(processed_data)   
        vocabulary_size = tokenization.vocabulary_size 
        logger.info(f'Vocabulary size (unique tokens): {vocabulary_size}')
        
        # split data into train set and validation set
        logger.info('Preparing dataset of images and captions based on splitting size')  
        splitter = TrainValidationSplit(self.configuration, processed_data)     
        train_data, validation_data = splitter.split_train_and_validation()        
               
        self.serializer.save_train_and_validation_data(
            train_data, validation_data, vocabulary_size) 
        logger.info('Preprocessed data saved into XREPORT database') 

    # define the logic to handle successfull data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_success(self, window, message):
        # send message to status bar
        window.statusBar().showMessage(message)
    
    # define the logic to handle error during data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_error(self, window, err_tb):
        exc, tb = err_tb
        QMessageBox.critical(window, 'Something went wrong!', f"{exc}\n\n{tb}")             



###############################################################################
class ValidationEvents:

    def __init__(self, configuration):        
        self.serializer = DataSerializer(configuration)            
        self.analyzer = ImageAnalysis(configuration)     
        self.configuration = configuration  

    #--------------------------------------------------------------------------
    def load_images_path(self, path, sample_size=1.0):        
        images_paths = self.serializer.get_images_path_from_directory(
            path, sample_size) 
        
        return images_paths 
        
    #--------------------------------------------------------------------------
    def run_dataset_evaluation_pipeline(self, metrics, progress_callback=None, worker=None):
        sample_size = self.configuration.get("sample_size", 1.0)
        images_paths = self.serializer.get_images_path_from_directory(IMG_PATH, sample_size)
        logger.info(f'The image dataset is composed of {len(images_paths)} images')            
               
        logger.info('Current metric: image dataset statistics')
        image_statistics = self.analyzer.calculate_image_statistics(
            images_paths, progress_callback, worker)                      

        images = []  
        if 'pixels_distribution' in metrics:
            logger.info('Current metric: pixel intensity distribution')
            images.append(self.analyzer.calculate_pixel_intensity_distribution(
                images_paths, progress_callback, worker))

        return images 

    #--------------------------------------------------------------------------
    def get_checkpoints_summary(self, progress_callback=None, worker=None): 
        summarizer = ModelEvaluationSummary(self.configuration)    
        checkpoints_summary = summarizer.get_checkpoints_summary(progress_callback, worker) 
        logger.info(f'Checkpoints summary has been created for {checkpoints_summary.shape[0]} models')   
    
    #--------------------------------------------------------------------------
    def run_model_evaluation_pipeline(self, metrics, selected_checkpoint, device='CPU', 
                                      progress_callback=None, worker=None):
        logger.info(f'Loading {selected_checkpoint} checkpoint from pretrained models')   
        modser = ModelSerializer()       
        model, train_config, session, checkpoint_path = modser.load_checkpoint(
            selected_checkpoint)    
        model.summary(expand_nested=True)  

        # set device for training operations based on user configuration
        logger.info('Setting device for training operations based on user configuration')                
        trainer = ModelTraining(train_config)    
        trainer.set_device(device_override=device)  

        # isolate the encoder from the autoencoder model   
        encoder = ImageEncoding(model, train_config, checkpoint_path)
        encoder_model = encoder.encoder_model 

        logger.info('Preparing dataset of images based on splitting sizes')  
        sample_size = train_config.get("train_sample_size", 1.0)
        images_paths = self.serializer.get_images_path_from_directory(IMG_PATH, sample_size)
        splitter = TrainValidationSplit(train_config) 
        _, validation_images = splitter.split_train_and_validation(images_paths)     

        # create the tf.datasets using the previously initialized generators 
        logger.info('Building model data loaders with prefetching and parallel processing') 
        # use tf.data.Dataset to build the model dataloader with a larger batch size
        # the dataset is built on top of the training and validation data
        loader = InferenceDataLoader(train_config)    
        validation_dataset = loader.build_inference_dataloader(
            validation_images, batch_size=1)              

        images = []
        if 'evaluation_report' in metrics:
            # evaluate model performance over the training and validation dataset 
            summarizer = ModelEvaluationSummary(self.configuration)       
            summarizer.get_evaluation_report(model, validation_dataset, worker=worker) 

        if 'image_reconstruction' in metrics:
            validator = ImageReconstruction(train_config, model, checkpoint_path)      
            images.append(validator.visualize_reconstructed_images(
                validation_images, progress_callback, worker=worker))       

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
        QMessageBox.critical(window, 'Something went wrong!', f"{exc}\n\n{tb}")  

   

###############################################################################
class ModelEvents:

    def __init__(self, configuration):        
        self.serializer = DataSerializer(configuration)        
        self.modser = ModelSerializer()         
        self.configuration = configuration 

    #--------------------------------------------------------------------------
    def get_available_checkpoints(self):
        return self.modser.scan_checkpoints_folder()
            
    #--------------------------------------------------------------------------
    def run_training_pipeline(self, progress_callback=None, worker=None):  
        logger.info('Preparing dataset of images based on splitting sizes')  
        sample_size = self.configuration.get("train_sample_size", 1.0)
        images_paths = self.serializer.get_images_path_from_directory(IMG_PATH, sample_size)

        splitter = TrainValidationSplit(self.configuration) 
        train_data, validation_data = splitter.split_train_and_validation(images_paths)
        
        # create the tf.datasets using the previously initialized generators 
        logger.info('Building model data loaders with prefetching and parallel processing')     
        builder = TrainingDataLoader(self.configuration)          
        train_dataset, validation_dataset = builder.build_training_dataloader(
            train_data, validation_data)
        
        # set device for training operations based on user configuration        
        logger.info('Setting device for training operations based on user configuration') 
        trainer = ModelTraining(self.configuration)           
        trainer.set_device()

        # build the autoencoder model 
        logger.info('Building FeXT AutoEncoder model based on user configuration') 
        checkpoint_path = self.modser.create_checkpoint_folder()
        autoencoder = FeXTAutoEncoder(self.configuration)           
        model = autoencoder.get_model(model_summary=True) 

        # check worker status to allow interruption
        check_thread_status(worker)   

        # generate training log report and graphviz plot for the model layout               
        self.modser.save_model_plot(model, checkpoint_path) 
        # perform training and save model at the end
        logger.info('Starting FeXT AutoEncoder training') 
        trainer.train_model(
            model, train_dataset, validation_dataset, checkpoint_path, 
            progress_callback=progress_callback, worker=worker)
        
    #--------------------------------------------------------------------------
    def resume_training_pipeline(self, selected_checkpoint, progress_callback=None, 
                                 worker=None):
        logger.info(f'Loading {selected_checkpoint} checkpoint from pretrained models')         
        model, train_config, session, checkpoint_path = self.modser.load_checkpoint(
            selected_checkpoint)    
        model.summary(expand_nested=True)  
        
        # set device for training operations based on user configuration
        logger.info('Setting device for training operations based on user configuration')         
        trainer = ModelTraining(self.configuration)           
        trainer.set_device()

        logger.info('Preparing dataset of images based on splitting sizes')  
        sample_size = train_config.get("train_sample_size", 1.0)
        images_paths = self.serializer.get_images_path_from_directory(IMG_PATH, sample_size)
        splitter = TrainValidationSplit(train_config) 
        train_data, validation_data = splitter.split_train_and_validation(images_paths)     

        # create the tf.datasets using the previously initialized generators 
        logger.info('Building model data loaders with prefetching and parallel processing') 
        builder = TrainingDataLoader(train_config)           
        train_dataset, validation_dataset = builder.build_training_dataloader(
            train_data, validation_data)  

        # check worker status to allow interruption
        check_thread_status(worker)         
                            
        # resume training from pretrained model    
        logger.info(f'Resuming training from checkpoint {selected_checkpoint}') 
        trainer.resume_training(
            model, train_dataset, validation_dataset, checkpoint_path, session,
            progress_callback=progress_callback, worker=worker)
        
    #--------------------------------------------------------------------------
    def run_inference_pipeline(self, selected_checkpoint, device='CPU', 
                               progress_callback=None, worker=None):
        logger.info(f'Loading {selected_checkpoint} checkpoint from pretrained models')         
        model, train_config, session, checkpoint_path = self.modser.load_checkpoint(
            selected_checkpoint)    
        model.summary(expand_nested=True)  

        # setting device for training         
        trainer = ModelTraining(train_config)    
        trainer.set_device(device_override=device)

        # select images from the inference folder and retrieve current paths        
        images_paths = self.serializer.get_images_path_from_directory(INFERENCE_INPUT_PATH)
        logger.info(f'{len(images_paths)} images have been found as inference input')  

        # check worker status to allow interruption
        check_thread_status(worker)   
             
        # extract features from images using the encoder output, the image encoder
        # takes the list of images path from inference as input    
        encoder = ImageEncoding(model, train_config, checkpoint_path)  
        logger.info(f'Start encoding images using model {selected_checkpoint}')  
        encoder.encode_images_features(images_paths, progress_callback, worker=worker) 
        logger.info('Encoded images have been saved as .npy')
           
        
    # define the logic to handle successfull data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_success(self, window, message):
        # send message to status bar
        window.statusBar().showMessage(message)
    
    # define the logic to handle error during data retrieval outside the main UI loop
    #--------------------------------------------------------------------------
    def handle_error(self, window, err_tb):
        exc, tb = err_tb
        QMessageBox.critical(window, 'Something went wrong!', f"{exc}\n\n{tb}")  

