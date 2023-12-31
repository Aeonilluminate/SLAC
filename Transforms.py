import numpy as np
import random
from scipy.ndimage import rotate
import Utility_Functions as UF





class Random_Rotate:

    def __init__(self, max_angle: int=360, order: int=1):
        self.max_angle = max_angle
        self.order     = order

        return None
    
    def __call__(self, image, helper):
        max_angle = self.max_angle
        angle     = np.random.uniform(low=0, high=max_angle)
        order     = self.order

        rotated_image = rotate(image, angle=angle, order=order, reshape=False)
        rotated_helper = rotate(helper, angle=angle, order=order, reshape=False)

        rotated_helper = (rotated_helper > 0.1).astype(int)

        return rotated_image, rotated_helper



class H_Reflect:
    
    def __init__(self):
        return None
        
    def __call__(self, img):
        reflected_img = img[:, ::-1]

        return reflected_img
    


class V_Reflect:

    def __init__(self):
        return None
    
    def __call__(self, image):
        reflected_image = image[::-1, :]

        return reflected_image



class Random_Shift:
    def __init__(self, y_shift_ratio_max=0.25, x_shift_ratio_max=0.25):
        self.y_shift_ratio_max = y_shift_ratio_max
        self.x_shift_ratio_max = x_shift_ratio_max

        return None
    
    def __call__(self, image, helper):
        y_shift_ratio_max = self.y_shift_ratio_max
        x_shift_ratio_max = self.x_shift_ratio_max

        y_shape, x_shape = image.shape[0:2]

        y_shift_limit = y_shape * y_shift_ratio_max
        x_shift_limit = x_shape * x_shift_ratio_max
        y_shift = int(random.uniform(-y_shift_limit, y_shift_limit))
        x_shift = int(random.uniform(-x_shift_limit, x_shift_limit))

        y_super_shape = y_shape + 2*abs(y_shift)
        x_super_shape = x_shape + 2*abs(x_shape)

        target_y_min = abs(y_shift) + y_shift
        target_x_min = abs(x_shift) + x_shift
        target_y_max = y_shape + target_y_min
        target_x_max = x_shape + target_x_min

        if len(image.shape) > 2:
            super_I = np.zeros((y_super_shape, x_super_shape, 3))
            super_H = np.zeros((y_super_shape, x_super_shape))
            super_I[target_y_min:target_y_max, target_x_min:target_x_max, :] = image[:, :, :]
            super_H[target_y_min:target_y_max, target_x_min:target_x_max] = helper[:, :]
        elif len(image.shape) == 2:
            super_I = np.zeros((y_super_shape, x_super_shape))
            super_H = np.zeros((y_super_shape, x_super_shape))
            super_I[target_y_min:target_y_max, target_x_min:target_x_max] = image[:, :]
            super_H[target_y_min:target_y_max, target_x_min:target_x_max] = helper[:, :]

        crop_y_min = abs(y_shift)
        crop_x_min = abs(x_shift)
        crop_y_max = y_shape + crop_y_min
        crop_x_max = x_shape + crop_x_min
        shifted_img = super_I[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
        shifted_helper = super_H[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        return shifted_img, shifted_helper
    


class Random_Patch:
    def __init__(self, max_num_patch: int=20, var_num_patch: bool=True, x_patch_ratio: float=0.05, y_patch_ratio: float=0.15,
                 x_patch_var: bool=True, y_patch_var: bool=True):
        self.max_num_patch = max_num_patch
        self.var_num_patch = var_num_patch
        self.x_patch_ratio = x_patch_ratio
        self.y_patch_ratio = y_patch_ratio
        self.x_patch_var = x_patch_var
        self.y_patch_var = y_patch_var

    def __call__(self, image, helper):
        x_patch_ratio = self.x_patch_ratio
        y_patch_ratio = self.y_patch_ratio

        mask = np.ones_like(image)

        if self.var_num_patch:
            num_patch = random.randint(int(self.max_num_patch/3), self.max_num_patch)
        else:
            num_patch = self.max_num_patch

        w = image.shape[1]
        h = image.shape[0]
        x_patch_max_size = int(w * x_patch_ratio)
        y_patch_max_size = int(h * y_patch_ratio)
        w_limit = w - x_patch_max_size
        h_limit = h - y_patch_max_size

        x_pos = np.random.randint(low=0, high=w_limit, size=num_patch)
        y_pos = np.random.randint(low=0, high=h_limit, size=num_patch)

        y_pos = y_pos.reshape(-1,1)
        x_pos = x_pos.reshape(-1,1)
        positions   = np.hstack((y_pos, x_pos))

        for (y, x) in positions:
            if self.x_patch_var:
                x_patch_size = random.randint(int(0.2 * x_patch_max_size), x_patch_max_size)
            if self.y_patch_var:
                y_patch_size = random.randint(int(0.2 * y_patch_max_size), y_patch_max_size)

            mask[y:y+y_patch_size, x:x+x_patch_size] = 0
        
        patched_image = image * mask

        return patched_image, helper
    

class Random_Noise:
    def __init__(self, max_num_lines=20, max_num_speckles=100, max_line_len_ratio=0.05, line_len_var=True):
        self.max_num_lines = max_num_lines
        self.max_num_speckles = max_num_speckles
        self.len_upper = max_line_len_ratio
        self.line_len_var = line_len_var

    def __call__(self, image, helper):
        num_lines    = random.randint(int(self.max_num_lines/3), self.max_num_lines)
        num_speckles = random.randint(int(self.max_num_speckles/3), self.max_num_speckles)
        w = image.shape[1]
        h = image.shape[0]
        d = max(w, h)
        bounding_len = self.len_upper*d
        for _ in range(num_lines):
            x1, y1 = np.random.randint(0, w - bounding_len), np.random.randint(0, h - bounding_len)
            x2 = np.random.randint(x1 + 1, int(x1 + bounding_len))
            y2_range = np.sqrt(bounding_len ** 2 - (x2 - x1) **2)
            y2 = np.random.randint(y1 + 1, y1 + y2_range)
            points = UF.bresenham_line(x1, y1, x2, y2)
            for (x, y) in points:
                if 0<= x < w and 0 <= y < h:
                    image[y, x] = 1
        
        for _ in range(num_speckles):
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            image[y, x] = 1

        return image, helper