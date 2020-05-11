from os import listdir
import string
from pickle import dump

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

# feature extraction
def feature_extraction(directory):
	model = VGG16()
 	# model = VGG19()
	# model = InceptionV3()
	# model = ResNet50()
	# model = InceptionResNetV2()
	# feature_extractor = InceptionResNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
	# model = feature_extractor.output
	# model = GlobalAveragePooling2D()(model)
	# model = Dropout(0.5)(model)
	# model = Dense(4096, activation="relu")(model)
	# model = Dropout(0.5)(model)
	# model = Dense(4096, activation="relu")(model)

	model.layers.pop()
	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

	features = dict()
	for name in listdir(directory):
		img = load_img(directory + '/' + name, target_size=(224, 224))
		img = img_to_array(img)
		img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
		img_id = name.split('.')[0]
		features[img_id] = model.predict(preprocess_input(img), verbose=0)
	return features

# extracting features and dumping the features into a pickle file
directory = 'Flicker8k_Dataset'
features = feature_extraction(directory)
print('Number of Features Extracted:', len(features))
dump(features, open('features.pkl', 'wb'))