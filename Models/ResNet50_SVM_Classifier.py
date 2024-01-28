import os
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn import metrics
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold, KFold
import warnings
import itertools
from itertools import cycle
from scipy import interp
import shutil
import random
import glob
import time
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam,Adadelta
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow
from sklearn.preprocessing import LabelEncoder
from random import randint
import sys


img_width, img_height = None  # TODO: Specify img_width, img_height
epochs = None  # TODO: Specify epochs
INIT_LR = None  # TODO: Specify INIT_LR
input_shape = None  # TODO: Specify input_shape

# Where to save model
filepath1 = None  # TODO: Specify filepath1

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
disable_eager_execution()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

warnings.simplefilter(action='ignore', category=FutureWarning)

train_data_dir = 'rwbc_datasets/train'
test_data_dir = 'rwbc_datasets/test'


num_classes = 19
FREEZE_LAYERS = 2


if K.image_data_format() == 'channels_first':
  input_shape = (3, img_width, img_height)
else:
  input_shape = (img_width, img_height, 3)

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 45, # random rotation btw 0 t0 45
    shear_range = 0.3,
    zoom_range = 0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_batches = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_width, img_height),
    shuffle=True,
    batch_size = None,  # TODO: Specify batch_size
    class_mode = 'categorical')


test_batches = test_datagen.flow_from_directory(
    test_data_dir,
    target_size = (img_width, img_height),
    batch_size = None,  # TODO: Specify batch_size
    class_mode = 'categorical')


for image_batch, labels_batch in test_batches:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

#Model Definition
def create_model():
    
    pretrained = ResNet50(include_top=False, weights='imagenet',input_shape=[img_width, img_height, 3])

    for layer in pretrained.layers:
        layer.trainable = None  # TODO: Specify True or False
            
    x = pretrained.output
    x = tensorflow.keras.layers.GlobalAveragePooling2D() (x)
    x = tensorflow.keras.layers.Flatten()(x)
    x = tensorflow.keras.layers.Dense(512, activation = 'relu')(x)
    x = tensorflow.keras.layers.Dropout(0.5)(x)
    x = tensorflow.keras.layers.Dense(units=128, activation='relu')(x)
    x = tensorflow.keras.layers.Dropout(0.5)(x)

    outputs = tensorflow.keras.layers.Dense((num_classes),kernel_regularizer=tf.keras.regularizers.l2(0.01), activation="softmax", dtype='float32')(x)
        
    model1 = tensorflow.keras.Model(pretrained.input, outputs)
    return model1

model1 = create_model()

# Model Compiling
model1.compile(optimizer=Adam(learning_rate = INIT_LR),
              loss="squared_hinge",
              metrics=['accuracy'])
model1.summary()

class_names=['01_Band_neutrophil', '02_Segmented_neutrophil', '03_Eosinophil', '04_Basophil', '05_Lymphocyte', '06_Monocyte', 
                      '07_Promyelocyte', '08_Myelocyte', '09_Metamyelocyte', '10_Prolymphocyte', '11_Immature_cell', '12_Lymphocyte_variant_form', 
                      '13_Plasma_cell', '14_Large_granular_lymphocyte', '15_Abnormal_cell', '16_Smudge_cell', '17_Artifact', '18_nRBC', '19_Giant_platelet']

# Prepare training data
x, y = next(train_batches)
new_ytrain = []
for i in range(len(y)):
    data = list(y[i])
    index_ = data.index(1)
    new_ytrain.append(index_)
yy = np.array(new_ytrain)

le = LabelEncoder()
y_encoder = le.fit_transform(yy.flatten())
y_train = y_encoder[:int(x.shape[0])]

x_train = x

# Prepare test data
x1, y1 = next(test_batches)
new_ytest = []
for i in range(len(y1)):
    data = list(y1[i])
    index_ = data.index(1)
    new_ytest.append(index_)
yy1 = np.array(new_ytest)

y_encoder = le.fit_transform(yy1.flatten())
y_test = y_encoder[:int(x1.shape[0])]

x_test = x1


# Training, Validating, and Testing loop for number of folds
def func_cv(X_train, y_train, cv_itr, n_cv):
    c_matrix_test = np.zeros((num_classes, num_classes), dtype=int)
    cm_agg = np.zeros((num_classes, num_classes), dtype=int)

    res_acc = np.zeros(n_cv)
    res_pre = np.zeros(n_cv)
    res_rec = np.zeros(n_cv)
    res_f1s = np.zeros(n_cv)
    
    res_acc_ts = np.zeros(n_cv)
    res_pre_ts = np.zeros(n_cv)
    res_rec_ts = np.zeros(n_cv)
    res_f1s_ts = np.zeros(n_cv)
    
    
    ix = 0
    
    for train_ix, test_ix in cv_itr.split(X_train, y_train):
        x_tr, y_tr = X_train[train_ix], y_train[train_ix]
        x_val, y_val = X_train[test_ix], y_train[test_ix]      
        
        history = model1.fit(x_tr, y_tr,
                            validation_data=(x_val, y_val),
                            epochs= epochs,
                            callbacks=[
                                tf.keras.callbacks.ReduceLROnPlateau(
                                    monitor = 'val_accuracy',
                                    factor = 0.2,
                                    patience = 10,
                                    verbose = 1,
                                    min_delta = 1e-4,
                                    min_lr = 1e-6,
                                    mode = 'max'),

                                tf.keras.callbacks.EarlyStopping(
                                    monitor = 'val_accuracy',
                                    min_delta = 1e-4,
                                    patience = 25,
                                    mode = 'max',
                                    restore_best_weights = True,
                                    verbose = 1),

                                tf.keras.callbacks.ModelCheckpoint(
                                    filepath = filepath1,
                                    monitor = 'val_accuracy', 
                                    verbose = 1, 
                                    save_best_only = True,
                                    save_weights_only = False,
                                    mode = 'max')
                            ])
        
        print("[INFO] saving model...")
        
        # load a saved model
        saved_model1 = tf.keras.models.load_model(
            filepath1,
            custom_objects=None,
            compile=True,
            options=None
        )


        # START ---TRAINING AND VALIDATION PART---
        y_pred_ = saved_model1.predict(x_val)        
        y_pred1 = []
        y_pred2 = []
        
        
        for i in y_pred_:
            y_pred1.append(np.argmax(i)) 
        for j in y_val:
            y_pred2.append(np.argmax(j))

        y_pred = np.array(y_pred1)
        y_pred_ts = np.array(y_pred2)
               
        
        print('\n\n') 
        print('-'* 40)
        print(time.strftime('%X %x %Z'))
        print('{0}-CV Performance Metrics'.format(ix))
        print('-'* 40)
        
        
        res_acc[ix] = cr['accuracy']
        res_pre[ix] = cr['macro avg']['precision']
        res_rec[ix] = cr['macro avg']['recall']
        res_f1s[ix] = cr['macro avg']['f1-score']
        
        
        print('\n\n')
        print('-'* 80)
        print('Validation')
        print('Accuracy:\t\t {0:.5f}'.format(cr['accuracy']))
        print('Avg precision:\t\t {0:.5f}'.format(cr['macro avg']['precision']))
        print('Avg recall:\t\t {0:.5f}'.format(cr['macro avg']['recall']))
        print('Avg f1-score:\t\t {0:.5f}'.format(cr['macro avg']['f1-score']))
          # END ---TRAINING AND VALIDATION PART---     

        
        # START ---TESTING PART---   
        i = randint(2, 9)       
        num_val = len(x_t)//i
        x_test_ = x_t[num_val:(i+1)*num_val]
        y_test_ = y1[num_val:(i+1)*num_val]
        
        y_pred_test = saved_model1.predict(x_test_) 
        
        y_pred_test1 = []
        y_pred_test2 = []
        
        for i in y_pred_test:
            y_pred_test1.append(np.argmax(i)) 
        for j in y_test_:
            y_pred_test2.append(np.argmax(j))
            
        y_pred_t = np.array(y_pred_test1)
        y_pred_tt = np.array(y_pred_test2)
        

        # Testing Classification
        c_matrix_test = confusion_matrix(y_pred_tt, y_pred_t)
       
        cm_agg = np.add(cm_agg, c_matrix_test) 
        
        #Testing
        res_acc_ts[ix] = cr_test['accuracy']
        res_pre_ts[ix] = cr_test['macro avg']['precision']
        res_rec_ts[ix] = cr_test['macro avg']['recall']
        res_f1s_ts[ix] = cr_test['macro avg']['f1-score']
        
        
        print('\n\n')
        print('Testing')
        print('Accuracy_test:\t\t\t {0:.5f}'.format(cr_test['accuracy']))
        print('Avg precision_test:\t\t {0:.5f}'.format(cr_test['macro avg']['precision']))
        print('Avg recall_test:\t\t {0:.5f}'.format(cr_test['macro avg']['recall']))
        print('Avg f1-score_test:\t\t {0:.5f}'.format(cr_test['macro avg']['f1-score']))


        print("")
        print("")
        print('-'* 80)
        ix = ix + 1  
        
        
        
    print('Aggregated metrics performance for Validation ')  
    print('-'* 80)
    print('Aggregated avg accuracy:\t {0:.5f}'.format(np.mean(res_acc)))
    print('Aggregated avg precision:\t {0:.5f}'.format(np.mean(res_pre)))
    print('Aggregated avg recall:\t\t {0:.5f}'.format(np.mean(res_rec)))
    print('Aggregated avg f1-score:\t {0:.5f}'.format(np.mean(res_f1s)))
    print('')
    
    print('-'* 80)
    print('Aggregated metrics performance for Testing')
    print('-'* 80)
    print('Aggregated AVG Accuracy_test:\t {0:.5f}'.format(np.mean(res_acc_ts)))
    print('Aggregated AVG Precision_test:\t {0:.5f}'.format(np.mean(res_pre_ts)))
    print('Aggregated AVG Recall_test:\t {0:.5f}'.format(np.mean(res_rec_ts)))
    print('Aggregated AVG f1-Score_test:\t {0:.5f}'.format(np.mean(res_f1s_ts)))
    print('')
    print('')
    print("")
    print('Normalized confusion_matrix')
    print('')

    def plot_confusion_matrix(cm_agg, classes,
                        normalize=True,
                        title='Normalized confusion matrix',
                        cmap=plt.cm.Blues):
                        
        plt.imshow(cm_agg, interpolation='nearest', cmap=cmap)
        plt.title(title)
        # plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm_agg = np.around(cm_agg.astype('float') / cm_agg.sum(axis=1)[:, np.newaxis], decimals= 2)
            print()
            # print("Normalized confusion matrix")
        else:
            print()
            print('Confusion matrix, without normalization')

        thresh = cm_agg.max() / 2.
        for i, j in itertools.product(range(cm_agg.shape[0]), range(cm_agg.shape[1])):
            plt.text(j, i, cm_agg[i, j],
                horizontalalignment="center",
                color="black" if cm_agg[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


    cm_plot_labels = ['01_Band_neutrophil', '02_Segmented_neutrophil', '03_Eosinophil', '04_Basophil', '05_Lymphocyte', '06_Monocyte', 
                      '07_Promyelocyte', '08_Myelocyte', '09_Metamyelocyte', '10_Prolymphocyte', '11_Immature_cell', '12_Lymphocyte_variant_form', 
                      '13_Plasma_cell', '14_Large_granular_lymphocyte', '15_Abnormal_cell', '16_Smudge_cell', '17_Artifact', '18_nRBC', '19_Giant_platelet']


    fig, ax = plt.subplots(figsize=(15, 15))
    plot_confusion_matrix(cm_agg=cm_agg, classes=cm_plot_labels, title='Normalized Confusion_matrix')



n_cross_valid = 10
kf = KFold(n_splits=n_cross_valid, shuffle=True, random_state=33)

print('\n')
print('*'*80)
print('*'*80)
print("ResNet50_SVM_Classifier")
print(time.strftime('%X %x %Z'))

print('*'*80)
func_cv(x_train,y, kf, n_cross_valid)