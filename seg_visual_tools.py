import json
import os
import cv2
import numpy as np
import math

from PIL import Image

# test
OUT_DIR = 'count_label/label_check'

# COLOR TUPLES (B,G,R) index0 is background
# you can also define your default color-map here
# CMAP = [(0, 0, 0), (255, 127, 0), (0, 255, 255), (255, 0, 127), (0, 191, 255), (0, 255, 64), (255, 255, 0),
#         (204, 204, 204)]

CMAP = [(0, 0, 0), (255, 127, 0), (0, 255, 254), (255, 0, 127), (0, 92, 230), (0, 255, 64), (179, 177, 0),
        (204, 204, 204)]

# CMAP =  [(0,0,0),(0,255,0),(0,0,255),(255,0,0),(250,170,30),(220,220,0),(107,142,35),(231,241,0)]


def load_array_from_png(img_path, mode='image'):
    if mode == 'mask':
        im_mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        arr_mask = np.array(im_mask)
        return arr_mask
    elif mode == 'image':
        im_img = cv2.imread(img_path)
        arr_img = np.array(im_img)
        return arr_img
    else:
        raise Exception('Invalid mode')


def load_points_from_json(json_path, case_name, kept_class):
    with open(json_path, 'r', encoding='utf-8') as load_f:
        load_dict = json.load(load_f)
        labels = load_dict[case_name]
        # print(labels)

        key_points = []

        for cls_idx in kept_class:
            if str(cls_idx) in labels:
                # print(str(cls_idx), labels[str(cls_idx)])
                p = labels[str(cls_idx)]
                # use pixel location
                cx, cy = p['cx'], p['cy']
                key_points.append([cx, cy])
            else:
                key_points.append([0, 0])

        return key_points


def get_binary_mask_layers(mask, classes, num_class, use_morphology=True):
    """
        get mask layers filled with 0 and 1
        input:
         - mask: 1d mask with category form 0~N, 0 is background by default
         - classes: list of classes index, example:[1,4]
         - num_class: int of classes num(N)
         - use_morphology: use morphological operation or not,
            used for filter some noise/linear in mask map
    """
    # check classes length
    if (num_class < len(classes)):
        raise Exception(f'num_class map is not enough,'
                        f' declared {num_class} classes but has {len(classes)} list')

    binary_mask_layers = []
    # mask从大到小依次提取出来
    for i in range(1, num_class):
        class_idx = num_class - i
        # 转为二值图：thresh=二值筛选阈值 maxval=填充色
        _, binary = cv2.threshold(src=mask, thresh=class_idx - 1, maxval=1, type=cv2.THRESH_BINARY)
        # 去除mask二值图中的细小噪声/细线
        if use_morphology:
            kernel = np.ones((5, 5), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            binary_mask_layers.append(binary)
        # 去除当前类别的信息
        mask = (mask - binary * (class_idx))
    # 逆序恢复到正常顺序
    binary_mask_layers = binary_mask_layers[::-1]
    # binary_mask_layers should have N-1 layers as background 0 is excluded
    # example: if you need classes=[1,4], then will return 0th, 3rd layer
    return [binary_mask_layers[i - 1] for i in classes]


def get_colorful_contour(mask, required_classes, num_class, cmap, thickness=3):
    """
        input:
            image: image with channel [B,G,R]
            classes: list of classes index, example:[1,4]
            num_class: number of classes 0-N (include background 0)
            mask: category gray-scale mask, example: for a 3*3 pixel pic:[[0,0,1],[0,2,0],[4,0,3]]
            cmap: color map for categories 0-N, index=0 is regard background
    """
    h, w = mask.shape

    # 1 check cmap length
    if (num_class > len(cmap)):
        raise Exception(f'color map is not enough, need {num_class} colors but only has {len(cmap)}')

    # 2 get binary mask for every category index
    binary_mask_layers = get_binary_mask_layers(mask, required_classes, num_class)

    # 3 get the colorful contours layers
    contour_layers = []
    idx = 0
    for cls in required_classes:
        bina = binary_mask_layers[idx]
        contours, hierarchy = cv2.findContours(image=bina, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        contours = cv2.drawContours(image=np.zeros((h, w, 3)), contours=contours,
                                    contourIdx=-1, color=cmap[cls],
                                    thickness=thickness, lineType=cv2.FILLED)
        contour_layers.append(contours)
        idx += 1
    return contour_layers


"""
input:
    image: image with channel [B,G,R]
    mask: category gray-scale mask, example: for a 3*3 pixel pic:[[0,0,1],[0,2,0],[4,0,3]]
"""


def blend_with_contour_mask(image, mask, required_classes, num_class, blend_weight=1.0, cmap=CMAP, thickness=3):
    if 0 in required_classes:
        raise Exception('Please do not require background class.')
    if isinstance(image, np.uint8):
        image = image.astype(np.uint8)
    contour_layers = get_colorful_contour(mask, required_classes=required_classes, num_class=num_class,
                                          cmap=cmap, thickness=thickness)
    for contour in contour_layers:
        if contour.shape != image.shape:
            raise Exception(f'shape not equal compared image:{image.shape} and contour:{contour.shape}')
        image = cv2.addWeighted(image, 1, contour.astype(np.uint8), blend_weight, 0)
    return image


def get_colorful_filled(mask, required_classes, num_class, cmap):
    h, w = mask.shape

    # 1 check cmap length
    if (num_class > len(cmap)):
        raise Exception(f'color map is not enough, need {num_class} colors but only has {len(cmap)}')

    # 2 get binary mask for every category index
    binary_mask_layers = get_binary_mask_layers(mask, required_classes, num_class)

    # 3 get the colorful filled layers
    mask_layers = []
    idx = 0
    for cls in required_classes:
        bina = binary_mask_layers[idx].astype(np.uint8)
        layer = np.expand_dims(bina, axis=2).repeat(3, axis=2)
        layer = layer * np.asarray(cmap[cls])
        mask_layers.append(layer.astype(np.uint8))
        idx += 1
    return mask_layers


def blend_with_filled_mask(image, mask, required_classes, num_class, img_weight=1, mask_weight=0.5, gamma=0, cmap=CMAP):
    if 0 in required_classes:
        raise Exception('Please do not require background class.')
    if isinstance(image, np.uint8):
        image = image.astype(np.uint8)
    mask_layers = get_colorful_filled(mask, required_classes=required_classes, num_class=num_class, cmap=cmap)
    for layer in mask_layers:
        if layer.shape != image.shape:
            raise Exception(f'shape not equal compared image:{image.shape} and mask layer:{layer.shape}')
        image = cv2.addWeighted(image, img_weight, layer.astype(np.uint8), mask_weight, gamma)
    return image


def get_point(angle, d, base):
    angle = angle / 180.0 * math.pi
    _x, _y = math.cos(angle) * d, math.sin(angle) * d
    return [base[0] + _x, base[1] - _y]


""""
input:
    canvas: canvas to plot on
    base: location of the center point
"""


def plot_star(image, base, size=120, blend_weight=1.0, start_angle=90, edges=5, color=(255, 0, 0)):
    if base[0] == 0 and base[1] == 0:
        return image
    canvas = np.zeros_like(image)
    x = size
    y = x / (math.cos(0.2 * math.pi) + math.sin(0.2 * math.pi) / math.tan(0.1 * math.pi))
    points = []
    angle = 360 // edges // 2
    for i in range(edges):
        points.append(get_point(start_angle, x, base))
        start_angle -= angle
        points.append(get_point(start_angle, y, base))
        start_angle -= angle
    points = np.array([points], np.int32)
    canvas = cv2.fillPoly(canvas, points, color, cv2.LINE_AA)
    image = cv2.addWeighted(image, 1, canvas.astype(np.uint8), blend_weight, 0)
    return image


def blend_with_star_points(image, points, points_classes, size=20, blend_weight=1, cmap=CMAP):
    # points_classes should indicate the class_idx of points
    # example: [pointA, pointB, pointC] should align with [class1, class2, class3]
    if len(points) != len(points_classes):
        raise ValueError(f"points and classes must have the same number:"
                         f" {len(points)} points and {len(points_classes)} classes.")
    for point, points_class in zip(points, points_classes):
        image = plot_star(image, point, blend_weight=blend_weight,
                          size=size, color=cmap[points_class])
    return image


def blend_with_circle_points(image, points, points_classes, size=20, blend_weight=0.5, thickness=-1, cmap=CMAP):
    # points_classes should indicate the class_idx of points
    # example: [pointA, pointB, pointC] should align with [class1, class2, class3]
    if len(points) != len(points_classes):
        raise ValueError(f"points and classes must have the same number:"
                         f" {len(points)} points and {len(points_classes)} classes.")
    for point, points_class in zip(points, points_classes):
        if point[0] == 0 and point[0] == 0:
            continue
        img_h, img_w = image.shape[:2]
        circle_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        circle_img = cv2.circle(circle_img, center=point, radius=int(size),
                                color=cmap[points_class], thickness=thickness)
        image = cv2.addWeighted(image, 1, circle_img, blend_weight, 0)
    return image


if __name__ == '__main__':

    OUT_DIR = './img_label_ts'
    # mk = "Dataset110_SellaSegment/labelsTr/case_0031-517.png"
    # im = "Dataset110_SellaSegment/imagesTr/case_0031-517.png"
    cn = "Dataset110_SellaSegment/labelsTs_center_points.json"

    im_dir = "Dataset110_SellaSegment/imagesTs/"
    mk_dir = "Dataset110_SellaSegment/labelsTs/"

    # extract data from files
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    listdir_img = os.listdir(im_dir)
    listdir_mk = os.listdir(mk_dir)

    from tqdm import tqdm
    with tqdm(total=len(listdir_img)) as pbar:
        for im,mk in zip(listdir_img, listdir_mk):
            case_name = im.split('/')[-1].split('.')[0]
            arr_source_test = load_array_from_png(os.path.join(im_dir, im), mode='image')
            arr_mask_test = load_array_from_png(os.path.join(mk_dir, mk), mode='mask')
            key_points_test = load_points_from_json(cn, case_name, kept_class=[1, 2, 3, 4, 5, 6])

            if arr_mask_test is None or arr_source_test is None or key_points_test is None:
                raise Exception('Something went wrong.')

            arr_source_test = blend_with_contour_mask(image=arr_source_test, required_classes=[1, 2, 3, 4, 5, 6], num_class=7, mask=arr_mask_test,
                                            blend_weight=0.8)
            # image = blend_with_filled_mask(image=arr_source_test, required_classes=[1,2,3,4,5,6], num_class=7, mask=arr_mask_test, blend_weight=0.5)
            image = blend_with_star_points(image=arr_source_test, points=key_points_test, points_classes=[1,2,3,4,5,6], blend_weight=1, size=20)
            # image = blend_with_circle_points(image=image, points=key_points_test, points_classes=[1, 2, 3, 4, 5, 6], thickness=-1,
            #                                  blend_weight=0.3, size=image.shape[0] * 0.2)
            # image = blend_with_circle_points(image=image, points=key_points_test, points_classes=[1, 2, 3, 4, 5, 6], thickness=3,
            #                                  blend_weight=1, size=10)

            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Image.fromarray(image.astype(np.uint8))

            cv2.imwrite(os.path.join(OUT_DIR, f'{case_name}.png'), image)
            pbar.update(1)
