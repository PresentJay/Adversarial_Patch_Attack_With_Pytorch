from torchvision import models
from torchvision import datasets, transforms
import torch
import numpy as np
import cv2

def creatmodel(model_name):
	if model_name == "resnet50":
		model = models.resnet50(pretrained=True)
	elif model_name == "vgg16":
		model = models.vgg16(pretrained=True) 
	elif model_name == "alexnet":
		model = models.alexnet(pretrained=True)
	elif model_name == "densenet121":
		model = models.densenet121(pretrained=True)
	elif model_name == "mobilenet_v2":
		model = models.mobilenet_v2(pretrained=True)
	elif model_name == "inception_v3":
		model = models.inception_v3(pretrained=True)
	elif model_name == "googlenet":
		model = models.googlenet(pretrained=True)
	else:
		print("The model name was entered incorrectly.")

def image_test(model, img_path):
	transform = transforms.Compose([transforms.Resize(224)])
	img = datasets.ImageFolder(img_path, transform=transform)

	output = model(transform(img))
	_, pred = torch.topk(output, 3, dim=1, largest=True, sorted=True)
	print('Predicted:', pred)

def video_test(model):

	cap = cv2.VideoCapture(0)

	print('width :%d, height : %d' % (cap.get(3), cap.get(4)))
	transform = transforms.Compose([transforms.Resize(224)])

	while(True):
		ret, frame = cap.read()    # Read 결과와 frame

		if(ret) :
			output = model(transform(frame))
			_, pred = torch.topk(output, 3, dim=1, largest=True, sorted=True)

			print('Predicted:', pred)
			
			# display output
			cv2.imshow("Real-time classification", frame)
			if cv2.waitKey(1) == ord('q'):
				break
	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	model = creatmodel("resnet50")

	#image_test(model, '1981_859_adversarial.png')
	video_test(model)
