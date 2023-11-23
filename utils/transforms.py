import torchvision.transforms as transforms
import torch
standard_transforms = transforms.Compose([
	transforms.Resize((256,512)),
	transforms.ToTensor()]
)

training_img_transforms = transforms.Compose([
	transforms.Resize((256,512)),
	transforms.ToTensor(),
	transforms.Normalize(mean=(0.5, 0.5, 0.5) , std=(0.5, 0.5, 0.5))]
)

training_flow_transforms = transforms.Compose([
	transforms.Resize((256,512))
]

)

