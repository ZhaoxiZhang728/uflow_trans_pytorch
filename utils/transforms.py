import torchvision.transforms as transforms

standard_transforms = transforms.Compose([
	transforms.Resize((256,512)),
	transforms.ToTensor()
	])

training_transforms = transforms.Compose([
	transforms.Resize((256,512)),
	transforms.ToTensor()
	])

