import torch
import numpy as np
import cv2
from torchvision import transforms
from networks.u2net import U2NET, U2NETP
from utils.utils import parameters_read

class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image = sample['img']
        # {'img': image, 'annot': annot, 'filename': filename}
        return {'img': ((image.astype(np.float32) / 255.0 - self.mean) / self.std)}

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, img_size=512, use_offset=True, mean=48):
        self.img_size = img_size
        self.use_offset = use_offset
        self.mean = np.array(mean)

    def __call__(self, sample):
        image = sample['img']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        # print(image.shape, mask.shape)
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        # new_image = np.ones((self.img_size, self.img_size, 3)) * self.mean
        new_image = np.zeros((self.img_size, self.img_size, 3))

        if self.use_offset:
            offset_w = (self.img_size - resized_width) // 2
            offset_h = (self.img_size - resized_height) // 2
            new_image[offset_h:offset_h + resized_height, offset_w:offset_w + resized_width] = image


        else:
            new_image[0:resized_height, 0:resized_width] = image

        return {'img': torch.from_numpy(new_image).to(torch.float32)}

class predict_from_image(object):
    def __init__(self, sample, model, mean=[0.17593498438026792, 0.17593498438026792, 0.17593498438026792],
            std=[0.14129553477703288, 0.14129553477703288, 0.14129553477703288], img_size=256):

        self.sample = {'img':sample}

        self.mean = mean
        self.std = std
        self.img_size = img_size
        valid_transforms = [
            Normalizer(mean=self.mean, std=self.std),
            Resizer(self.img_size, mean=self.mean)
            ]
        self.predict_transform = transforms.Compose(valid_transforms)

        predict = self.predict_transform(self.sample)
        predict = predict['img'].cuda()
        predict = predict.permute(2, 0, 1)
        predict = torch.unsqueeze(predict, 0)

        output = model(predict).sigmoid()
        output[output>=0.5] = 1
        output[output<0.5] = 0
        seg_save = output.float()

        seg_save = seg_save.cpu().detach().numpy()
        self.seg_save = seg_save
    def get_image(self):
        return self.seg_save.astype(np.uint8)

def resize_2_original(h,w, pred):
    if h > w:
        scale = h / 128
        resized = int(128 * scale)
        crop = int((h - w) / 2)
        pred = cv2.resize(pred, (resized, resized), interpolation=cv2.INTER_NEAREST)
        pred = pred[:, crop:crop + w, :]
    else:
        scale = w / 128
        resized = int(128 * scale)
        crop = int((w - h) / 2)
        # print(crop)
        pred = cv2.resize(pred, (resized, resized), interpolation=cv2.INTER_NEAREST)
        pred = pred[crop:crop + h, :, :]
    return pred

def create_mask(pred):
    bg = pred.copy()
    bg = np.sum(bg, axis=2)
    bg = (bg == 0).astype(np.uint8)
    bg = np.expand_dims(bg, axis=2)
    mask = np.concatenate((bg, pred), axis=2)
    mask_decode = np.argmax(mask, axis=2)
    return mask_decode

def load_model(model_size, weight):
    params = parameters_read(f'configs/model_params/{model_size}.yml')
    # model = Segformer(**params.segformerparams)
    model = U2NETP(params.u2netparams['channels'], params.u2netparams['num_classes'])

    model.load_state_dict(weight, strict=False)
    model = model.cuda()
    model.eval()
    return model

weight = torch.load('./runs/u2net_trials_m9Ndx/cps/63_4928.pth')
model_name = "u2net"
model = load_model(model_name, weight)

img = cv2.imread("E:/Dataset/Carotid/dataset/Segmentation_B/dataset/val/image/image_24.png")
pred = predict_from_image(img, model).get_image()
pred = np.squeeze(pred)

h, w, _ = img.shape

pred = np.transpose(pred, (1,2,0))

pred = resize_2_original(h, w, pred)
mask_decode = create_mask(pred)

color_map = np.array([
    [0, 0, 0],
    [0, 0, 255],
    [0, 255, 255],
    [255, 0, 0],
])

mask = color_map[mask_decode].astype(np.uint8)

img_out = cv2.addWeighted(img, 0.5, mask, 0.5, 0)
cv2.imshow("test", img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()


