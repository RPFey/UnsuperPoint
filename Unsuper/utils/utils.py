import numpy as np
import collections
import random

def resize_img(img, IMAGE_SHAPE):
    h,w = img.shape[:2]
    if h < IMAGE_SHAPE[0] or w < IMAGE_SHAPE[1]:
        new_h = config.IMAGE_SHAPE[0]
        new_w = config.IMAGE_SHAPE[1]
        h = new_h
        w = new_w
        img = cv2.resize(img,(new_w, new_h))
    new_h, new_w = IMAGE_SHAPE
    try:
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
    except:
        print(h,new_h,w,new_w)
        raise
    if len(img.shape) == 2:
        img = img[top: top + new_h,left: left + new_w] # crop image
    else:
        img = img[top: top + new_h,left: left + new_w, :]
    return img

def dict_update(d, u):
    """Improved update for nested dictionaries.

    Arguments:
        d: The dictionary to be updated.
        u: The update dictionary.

    Returns:
        The updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def get_dst_point(perspective, IMAGE_SHAPE):
    a = random.random()
    b = random.random()
    c = random.random()
    d = random.random()
    e = random.random()
    f = random.random()

    if random.random() > 0.5:
        left_top_x = perspective*a
        left_top_y = perspective*b
        right_top_x = 0.9+perspective*c
        right_top_y = perspective*d
        left_bottom_x  = perspective*a
        left_bottom_y  = 0.9 + perspective*e
        right_bottom_x = 0.9 + perspective*c
        right_bottom_y = 0.9 + perspective*f
    else:
        left_top_x = perspective*a
        left_top_y = perspective*b
        right_top_x = 0.9+perspective*c
        right_top_y = perspective*d
        left_bottom_x  = perspective*e
        left_bottom_y  = 0.9 + perspective*b
        right_bottom_x = 0.9 + perspective*f
        right_bottom_y = 0.9 + perspective*d

    dst_point = np.array([(IMAGE_SHAPE[1]*left_top_x,IMAGE_SHAPE[0]*left_top_y,1),
            (IMAGE_SHAPE[1]*right_top_x, IMAGE_SHAPE[0]*right_top_y,1),
            (IMAGE_SHAPE[1]*left_bottom_x,IMAGE_SHAPE[0]*left_bottom_y,1),
            (IMAGE_SHAPE[1]*right_bottom_x,IMAGE_SHAPE[0]*right_bottom_y,1)],dtype = 'float32')
    return dst_point

def enhance(img, config):
    IMAGE_SHAPE = configp['IMAGE_SHAPE']
    seed = random.randint(1,20)
    src_point = np.array([[               0,                0],
                          [IMAGE_SHAPE[1]-1,                0],
                          [               0, IMAGE_SHAPE[0]-1],
                          [IMAGE_SHAPE[1]-1, IMAGE_SHAPE[0]-1]], dtype = np.float32)  # 圖片的四個頂點

    dst_point = get_dst_point(config['perspective'], IMAGE_SHAPE)
    center = (IMAGE_SHAPE[1]/2, IMAGE_SHAPE[0]/2)
    rot = random.randint(-2,2)*config['homographic']['rotation'] + random.randint(0,15)
    scale = 1.2 - config['homographic']['scale']*random.random()
    RS_mat = cv2.getRotationMatrix2D(center, rot, scale)
    f_point = np.matmul(dst_point, RS_mat.T).astype('float32')
    mat = cv2.getPerspectiveTransform(src_point, f_point)
    out_img = cv2.warpPerspective(img, mat, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]))
    if seed > 10 and seed <= 15:
        out_img = cv2.GaussianBlur(out_img, (3, 3), sigmaX=0)
    return out_img, mat

