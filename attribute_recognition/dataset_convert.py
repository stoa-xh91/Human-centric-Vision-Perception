import os
import glob
data_root = '/raid/wangxuanhan/datasets/PETA'
target_root = 'data/PETA/images'
count = 0
for f in os.listdir(data_root):
    source = os.path.join(data_root, f, 'archive/*')
    for image_name in glob.glob(source):
        target = image_name.split('/')[-1]
        target = os.path.join(target_root, target)
        count+=1
        print(count, image_name,target)
        os.system('ln -s {} {}'.format(image_name, target))

# source = os.path.join(lup1m_path, 'images')
# target = os.path.join(target_root, 'person')
# os.system('ln -s {} {}'.format(source, target))


