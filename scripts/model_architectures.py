import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet121

class ModelArchitectures():
    def __init__(self, class_names):
        self.base_model_1 = DenseNet121(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3),
            pooling='avg'             # Equivalent to layer.GlobalAveragePooling2D() in Sequential()
        )

        self.data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),  # ~10% rotation
            layers.RandomZoom(0.1),
        ])

        self.total_models = 3
        self.class_names = class_names

    def model_1(self):
        '''
        Base Model: DenseNet121 
            without top layer
            trained on current training set: False

        Data Augmentation: False
        '''
        self.base_model_1.trainable = False  # Freeze the base model 

        model = models.Sequential(
            [
                layers.Rescaling(1./255, input_shape=(224, 224, 3)),  # Normalize input
                self.base_model_1,
                # layers.GlobalAveragePooling2D(),                    # excluding because pooling = 'avg' in base_model_1
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(len(self.class_names), activation='softmax')  # Output layer
            ]
        )

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model
    
    def model_2(self):
        '''
        Base Model: DenseNet121 
            without top layer
            trained on current training set: False

        Data Augmentation: True
        '''        
        self.base_model_1.trainable = False  # Freeze the base model

        model = models.Sequential(
            [
                self.data_augmentation,
                layers.Rescaling(1./255, input_shape=(224, 224, 3)),  # Normalize input
                self.base_model_1,
                # layers.GlobalAveragePooling2D(),                    # excluding because pooling = 'avg' in base_model_1
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(len(self.class_names), activation='softmax')  # Output layer
            ]
        )

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def model_3(self):
        '''
        Base Model: DenseNet121 
            without top layer
            trained on current training set: True

        Data Augmentation: True
        '''

        self.base_model_1.trainable = True  # Un-Freeze the base model 

        model = models.Sequential(
            [
                self.data_augmentation,
                layers.Rescaling(1./255, input_shape=(224, 224, 3)),  # Normalize input
                self.base_model_1,
                # layers.GlobalAveragePooling2D(),                    # excluding because pooling = 'avg' in base_model_1
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.5),
                layers.Dense(len(self.class_names), activation='softmax')  # Output layer
            ]
        )

        model.compile(
            optimizer=Adam(1e-5),  # lower LR
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model
    
    def get_all_model_archs(self):
        return [self.model_1(), self.model_2(), self.model_3()]
            