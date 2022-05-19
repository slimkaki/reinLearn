import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

from tensorflow.python.client import device_lib 
print(device_lib.list_local_devices()) 

with tf.device("gpu:0"):
   print("tf.keras code in this scope will run on GPU")


#import keras
#config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
#sess = tf.Session(config=config) 
#keras.backend.set_session(sess)