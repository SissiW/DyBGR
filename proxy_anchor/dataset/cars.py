from .base import *
import scipy.io

class Cars(BaseDataset):
    def __init__(self, root, mode, transform = None):
        self.root = root #+ 'cars196'
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            self.classes = range(0,98)
        elif self.mode == 'eval':
            self.classes = range(98,196)
        # print('0', self.classes)  # range(0, 98)
        BaseDataset.__init__(self, self.root, self.mode, self.transform)

        annos_fn = 'car_ims/cars_annos.mat'
        cars = scipy.io.loadmat(os.path.join(self.root, annos_fn))
        # print('2', cars['annotations'][0])  # len=16185  [(array(img_path), array(bbox_x1), array(bbox_y1), array(bbox_x2), array(bbox_y2), array(class), array(test))]
        ys = [int(a[5][0] - 1) for a in cars['annotations'][0]]  # class_ID (0-195)
        im_paths = [a[0][0] for a in cars['annotations'][0]]

        index = 0
        for im_path, y in zip(im_paths, ys):
            # print(im_path, '----', y)  # im_path: car_ims/000089.jpg y: 0=class_ID
            if y in self.classes: # choose only specified classes
                self.im_paths.append(os.path.join(self.root, im_path))  # ['/home/hll/work/data/cars196/car_ims/000001.jpg', ...]
                self.ys.append(y)  # [0, 0, ..., 195]
                self.I += [index]  # [0,1,2, ..., 16184]
                index += 1