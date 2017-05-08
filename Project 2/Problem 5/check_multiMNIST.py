import numpy as np, cv2
from keras.models import load_model
# This script runs a model for two-digit MNIST and print the prediction of model.
#Change the image fine name if you would like to a different test file
img_file = './sample_images/img20005.jpg'

model = load_model('current_model_conv.h5')
im = cv2.imread(img_file, 0)
im = np.reshape(im, (1,1,42,28))
dig1, dig2 = model.predict(im)
print('digits=({},{})'.format(np.argmax(dig1), np.argmax(dig2)))