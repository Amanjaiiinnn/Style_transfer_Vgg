from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        super(ImageFolderDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.files = os.listdir(root)
        self.image_files = [p for p in self.files if p.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx): #image read and transform then return.
        img_path = os.path.join(self.root, self.image_files[idx])
        image = Image.open(img_path).convert('RGB') #dataset might contain black-whie  images . so .convert('RGB') used.

        if self.transform:
            image = self.transform(image)

        return image

def get_transform(size, crop, final_size): # create transform from library
    transform_list = []
    if size > 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.Resize(final_size))

    transform_list.append(transforms.ToTensor()) # final image to tensor
    return transforms.Compose(transform_list)    # create a transform list and return the composed transform    


def adaptive_instance_normalization(content_feat, style_feat):
    # [batch size, channels which is 2d, h, w] but input is 4d tensor
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)  #Extract statistics from content image.
    normalized_content_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)  
    return normalized_content_feat * style_std.expand(size) + style_mean.expand(size)  #Apply style statistics to normalized content features and return the transformed features. heart of AdaIN.

def calc_mean_std(feat, eps=1e-5):
    # [batch size, channels which is 2d, h, w] but input is 4d tensor
    size = feat.size()
    assert (len(size) == 4)
    batch_size, channels = size[:2]
    feat_mean = feat.view(batch_size, channels, -1).mean(dim=2).view(batch_size, channels, 1, 1 )  # channels, 1, 1 MAKE 4D TENSOR
    feat_var = feat.view(batch_size, channels, -1).var(dim=2, unbiased=False) + eps
    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1) #Because later broadcasting needs compatible dimensions.
    return feat_mean, feat_std    