from utils import  prepare_for_model
from keras.models import load_model,  Model
from layers import CoupledConvCapsule, CapsMaxPool, CapsuleNorm, CapsuleLayer, Length
from losses import margin_loss, seg_margin_loss
from keras import optimizers, metrics, activations
import cv2
import numpy as np 
from keras.preprocessing import image
import matplotlib.pyplot as plt 




model = load_model('out9.h5', custom_objects={
	'CoupledConvCapsule': CoupledConvCapsule,
	'CapsMaxPool': CapsMaxPool,
	'CapsuleNorm': CapsuleNorm,
	'CapsuleLayer': CapsuleLayer,
	'Length': Length,
	'_margin_loss': margin_loss(
		downweight=0.5, pos_margin=0.7, neg_margin=0.3
	),
	'_seg_margin_loss': seg_margin_loss(),
})


model=model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.0001, decay=1e-6),
              metrics=[metrics.categorical_accuracy])

img = image.load_img('test.jpg', target_size=None, color_mode='rgb')
#type(img)
x = image.img_to_array(img)
 
# plt.imshow(x)
# model.predict_classes(x)
# Model.predict(x, batch_size =1)
# Model.predict(x)

img = image.load_img()


# plt.matshow(x)
# # img = cv2.imread('test.jpg')
# # img = cv2.resize(img,(32,32))

#img = np.reshape(img,[1,320,240,3])

# # classes = model.predict_classes(img)

# # print (classes)model.predict_classes(x)

# #load_model('out9.h5')

