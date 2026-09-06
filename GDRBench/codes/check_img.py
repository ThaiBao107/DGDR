import os
from traceback import print_list
from create_mask import *

def get_image_filenames(folder_path, exts={'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}):
    image_filenames = []

    for root, dirs, files in os.walk(folder_path):  # Duyệt cả thư mục con
        for file in files:
            if os.path.splitext(file)[1].lower() in exts:
                image_filenames.append(os.path.join(root, file))

    return image_filenames

# Ví dụ dùng
folder_img = 'D:\\NCKH\\NCKH2024-2025\\Project\\Seg-set\\Original_Images'  # Thay đổi đường dẫn ở đây
image_ima = get_image_filenames(folder_img)

fouder_mask = 'D:\\NCKH\\NCKH2024-2025\\Project\\DGDR\\data\\FundusDG\\masks\\FGADR\\severe_npdr'
image_masks = get_image_filenames(fouder_mask)


list_img = []
list_mask = []
# In ra
for path in image_ima:
    x = path.split('\\')[-1]
    list_img.append(x)



for path in image_masks:
    x = path.split('\\')[-1]
    list_mask.append(x)


list_result = []
for i in list_img:
    if i in list_mask:
        list_result.append(i)


print(list_result)
print("Len: ", len(list_result))
#
# create_mask(list_result)
# print('done')


