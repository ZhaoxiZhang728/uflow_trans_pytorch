
from torch.utils.data import Dataset
import torch
class Video_dataset(Dataset):
    def __init__(self, frames,frame_transform=None):
        '''

        :param root: path to mp4
        :param frame_transform: to transform the images

        '''
        super(Video_dataset).__init__()

        self.frame_transform = frame_transform
        self.frames = frames


        self.n = len(self.frames)

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        # so if num is equal to 1, it will be compared to itself
        # randomly generate several images index of frames
        #sequence = self.index_to_sequence(index)
        # for training, randomly sample pairs from sequence
        #fn = np.random.choice(sequence, size=2)
        im1 = self.frames[index]
        im2 = self.frames[index-1]
        if self.frame_transform:
            im1 = self.frame_transform(im1)
            im2 = self.frame_transform(im2)

        dicts = {
            'images':torch.stack([im1,im2]) # the output size after dataloader is [batch_size, 2, 3,height,width]
            ,
            'labels':{
                'flow_uv': False,
                'flow_valid':False,
                'occlusions':False,
                'images_without_photo_aug':False
            }
        }
        return dicts



if __name__ == '__main__':
    # demo example
    ds = Video_dataset('../files/billiard_clip.mp4')
    print('ds is done')
    #dl = DataLoader(dataset = ds, shuffle=False,batch_size=1,num_workers=5)
