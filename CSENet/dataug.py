import albumentations as A
import cv2
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2


# 用来获取所有图形的均值和方差的transform
get_mean_std_transform = A.Compose([
        A.transforms.ToFloat(max_value=255),
        ToTensorV2(),
    ])



def transform(mean, std, size, transforme_p):
    """

    :param mean: 均值
    :param std: 方差
    :param size: 裁剪尺寸列表（epoch）
    :param transforme_p: 图像增强概率列表（epoch）
    :return: 训练数据，验证数据，测试数据
    """

    train_transform = A.Compose(
        [
         # A.Resize(540, 540),
         # A.RandomCrop(600,600),
         A.GaussNoise( p=transforme_p),
         # # A.Resize(size, size),
         # A.CoarseDropout(max_holes=int((500/7)**2*0.75), max_height=7, max_width=7, p=transforme_p),
         A.Flip(p=transforme_p),
         A.RandomRotate90(p=transforme_p),
         # # A.ElasticTransform(p=transforme_p, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
         # # A.GridDistortion(p=transforme_p),
         # # A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=transforme_p),
         # # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=transforme_p),
         # A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=transforme_p),
         # A.RandomBrightnessContrast(brightness_limit=0.6, contrast_limit=0.2, p=transforme_p),
         # A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=3, shadow_roi=(0, 0, 1, 1), p=transforme_p),
         A.Normalize(mean=mean, std=std),
         ToTensorV2(),

        ]
    )

    val_transform = A.Compose(
        [A.Normalize(mean=mean, std=std), ToTensorV2()]
    )
    test_transform = A.Compose(
        [A.Normalize(mean=mean, std=std), ToTensorV2()]
    )
    return train_transform,  val_transform, test_transform


# transforme_p = 1
# train_transform = A.Compose(
#     [
#         # A.Resize(540, 540),
#         # A.RandomCrop(512,512),
#         A.GaussNoise( p=transforme_p),
#         # A.Resize(size, size),
#         A.CoarseDropout(max_holes=int((500/7)**2*0.75), max_height=7, max_width=7, p=transforme_p),
#         A.Flip(p=transforme_p),
#         A.RandomRotate90(p=transforme_p),
#         # A.Transpose(p=transforme_p),
#         # A.ElasticTransform(p=transforme_p, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
#         # A.GridDistortion(p=transforme_p),
#         # A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=transforme_p),
#         # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=transforme_p),
#         # A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=transforme_p),
#         # A.RandomBrightnessContrast(brightness_limit=0.6, contrast_limit=0.2, p=transforme_p),
#         # A.RandomShadow(num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=3, shadow_roi=(0, 0, 1, 1),
#         #                p=transforme_p),
#
#     ]
# )
# from cv2 import imread, COLOR_BGR2RGB, IMREAD_GRAYSCALE, cvtColor
# image = imread('Massachusetts_road/images/testing/4.png')
# label = imread('Massachusetts_road/annotations/testing/4.png')
#
#
#
#
# image = train_transform(image=image, mask=label)
# label = image['mask']
# image = image['image']
#
#
# plt.subplot(121)
# plt.imshow(image)
# plt.subplot(122)
# plt.imshow(label)
# plt.show()