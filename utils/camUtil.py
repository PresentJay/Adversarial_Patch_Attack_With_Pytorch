from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import cv2

def image_test(model, img_path):
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)

	preds = model.predict(x)
	# decode the results into a list of tuples (class, description, probability)
	# (one such list for each sample in the batch)
	print('Predicted:', decode_predictions(preds, top=3)[0])

def video_test(model):

	cap = cv2.VideoCapture(0)

	print('width :%d, height : %d' % (cap.get(3), cap.get(4)))

	while(True):
		ret, frame = cap.read()    # Read 결과와 frame

		if(ret) :
			img = image.smart_resize(frame, (224,224), interpolation='bilinear')
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)

			preds = model.predict(x)
			print('Predicted:', decode_predictions(preds, top=3)[0])
			
			# display output
			cv2.imshow("Real-time object detection", frame)
			if cv2.waitKey(1) == ord('q'):
				break
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	model = ResNet50(weights='imagenet')

	#image_test(model, '1981_859_adversarial.png')
	video_test(model)
