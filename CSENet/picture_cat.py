import os
from PIL import Image
from random import sample, choices

dir_list = os.listdir('result')

dir_list = [dir_list[3], dir_list[0], dir_list[4], dir_list[2], dir_list[1], dir_list[5]]
dir_list.reverse()
print(dir_list)

# 将预测图片转到images
for i in range(680):
    s = []
    for dir in dir_list:
        if dir !='images':
            dir_path = os.path.join('result', dir)
            image_path = os.path.join(dir_path, 'test_images')
            pre_path_name = os.path.join(image_path, str(i)+'pre.png')
            s.append(pre_path_name)
    image_path_name = os.path.join(image_path, str(i) + 'image.png')
    mask_path_name = os.path.join(image_path, str(i) + 'mask.png')
    s.append(mask_path_name)
    s.append(image_path_name)
    s.reverse()
    print(s)
    name = ['image', 'mask', 'pool4_4', 'pool4', 'UNet', 'upernet', 'deeplabv3plus', 'None']
    COL = 4  # 指定拼接图片的列数
    ROW = 2  # 指定拼接图片的行数
    UNIT_HEIGHT_SIZE = 512  # 图片高度
    UNIT_WIDTH_SIZE = 512  # 图片宽度
    PATH = s  # 需要拼接的图片所在的路径
    NAME = name  # 拼接出的图片保存的名字
    RANDOM_SELECT = False  # 设置是否可重复抽取图片
    SAVE_QUALITY = 50  # 保存的图片的质量 可选0-100

    for index in range(COL * ROW):
        image_files = [Image.open(x) for x in PATH]  # 读取所有用于拼接的图片
        if index == 7:
            image_files.append(Image.new('RGB', [512, 512], 0))

    target = Image.new('RGB', (UNIT_WIDTH_SIZE * COL, UNIT_HEIGHT_SIZE * ROW))  # 创建成品图的画布
    # 第一个参数RGB表示创建RGB彩色图，第二个参数传入元组指定图片大小，第三个参数可指定颜色，默认为黑色
    for row in range(ROW):
        for col in range(COL):
            # 对图片进行逐行拼接
            # paste方法第一个参数指定需要拼接的图片，第二个参数为二元元组（指定复制位置的左上角坐标）
            # 或四元元组（指定复制位置的左上角和右下角坐标）
            target.paste(image_files[COL * row + col], (0 + UNIT_WIDTH_SIZE * col, 0 + UNIT_HEIGHT_SIZE * row))
    new_name = str(i)+'.png'
    target.save(os.path.join('RESULT/images', new_name), quality=SAVE_QUALITY)  # 成品图保存


