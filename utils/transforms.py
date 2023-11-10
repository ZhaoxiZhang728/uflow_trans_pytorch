import torchvision.transforms as transforms

standard_transforms = transforms.Compose([
	transforms.ToTensor(),
	transforms.Resize((256,512))
	])

training_img_transforms = transforms.Compose([
	transforms.Resize((256,512)),
	transforms.ToTensor()
	])

training_flow_transforms = transforms.Compose([
	transforms.Resize((256,512))
	])

