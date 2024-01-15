import os

out_path = '/home/oq55olys/Projects/detection/OpenPCDet/data/kitti/ImageSetsRaw/'

file_path = '/home/oq55olys/Projects/detection/OpenPCDet/data/kitti/training/label_2/'

test_file_path = '/home/oq55olys/Projects/detection/OpenPCDet/data/kitti/training_old/label_2/'

#get all filenames with .txt extension
files = [f for f in os.listdir(file_path) if f.endswith('.txt')]

#write them to val.txt

#delete val.txt if it exists
if os.path.exists(out_path + 'val.txt'):
    os.remove(out_path + 'val.txt')
if os.path.exists(out_path + 'train.txt'):
    os.remove(out_path + 'train.txt')
if os.path.exists(out_path + 'test.txt'):    
    os.remove(out_path + 'test.txt')

with open(out_path + 'val.txt', 'w') as f:
    for item in files:
        f.write("%s\n" % item[:-4])

#write them to train.txt
with open(out_path + 'train.txt', 'w') as f:
    for item in files:
        f.write("%s\n" % item[:-4])


files = [f for f in os.listdir(test_file_path) if f.endswith('.txt')]

#write them to test.txt
with open(out_path + 'test.txt', 'w') as f:
    for item in files:
        f.write("%s\n" % item[:-4])
