import os
import numpy as np

import parser
#import open3d as o3d

def inverse_rigid_trans(Tr):
    """ Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    """
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr

#remove bounding box labels if there are no points in the bounding box
#format is KITTI
def remove_empty_labels (path, bbc):


    label_path = path + 'label_raw/'
    lidar_path = path + 'velodyne/'
    calib_path = path + 'calib/'

    #for set_path: add /ImageSets/ before the second last /
    #divide the path by / and get the second last element
    set_path = path.split('/')
    #read from the second last element if it is train or test mode
    mode = set_path[-2]
    #remove the training/ or testing/ at the end
    set_path = set_path[:-2]
    #add ImageSets
    set_path.append('ImageSetsRaw/')
    #make it a string
    set_path = '/'.join(set_path)
   

    print(set_path)


    set_path_out = set_path.replace('Raw', '')

    label_out_path = path + 'label_2/'
    #make the output directory if it doesn't exist
    if not os.path.exists(label_out_path):
        os.makedirs(label_out_path)


    label_files = [f for f in os.listdir(label_path) if f.endswith('.txt')]
    lidar_files = [f for f in os.listdir(lidar_path) if f.endswith('.bin')]
    calib_files = [f for f in os.listdir(calib_path) if f.endswith('.txt')]



    #zip the files together
    files = zip(label_files, lidar_files, calib_files)
    for label_file, lidar_file, calib_file in files:
        #read in the label file
        with open(label_path + label_file, 'r') as f:
            lines = f.readlines()
        #read in the lidar file
        pointcloud = np.fromfile(lidar_path + lidar_file, dtype=np.float32, count=-1).reshape([-1, 4])
        #read in the calib file
        with open(calib_path + calib_file, 'r') as f:
            c_lines = f.readlines()
            #get rectification matrix
            R0_rect = np.array(c_lines[4].split(' ')[1:], dtype=np.float32).reshape([3, 3])
            #make homogenous
            R0_rect = np.vstack((R0_rect, np.array([0, 0, 0])))
            R0_rect = np.hstack((R0_rect, np.array([[0], [0], [0], [1]])))
               

            #get velodyne to camera matrix
            Tr_velo_to_cam = np.array(c_lines[5].split(' ')[1:], dtype=np.float32).reshape([3, 4])



  
        #get the bounding box labels
        lc = 0
        car_corners = np.array([])
        for line in lines:
            line = line.split(' ')
            if line[0] == 'Car':
                #get the 3d bounding box
                h = float(line[8])
                w = float(line[9])
                l = float(line[10])
           
                #get bounding box points but add position only in the end after my computations
                x_min = -l / 2
                x_max = l / 2
                y_min = 0
                y_max =0- h
                z_min = -w / 2
                z_max = w / 2
                
                corners = np.array([[x_min, y_min, z_min],
                                    [x_min, y_min, z_max],
                                    [x_min, y_max, z_min],
                                    [x_min, y_max, z_max],
                                    [x_max, y_min, z_min],
                                    [x_max, y_min, z_max],
                                    [x_max, y_max, z_min],
                                    [x_max, y_max, z_max]])
                
       
                #get rotation
                rot_y = float(line[14])
                
                #rotate the corners
                corners = np.hstack((corners, np.ones((corners.shape[0], 1))))
       

                #this assumes taht rot_y is in range [-pi, pi]

                rot_y_matrix = np.array([[np.cos(rot_y), 0, np.sin(rot_y), 0],
                                        [0, 1, 0, 0],
                                        [-np.sin(rot_y), 0, np.cos(rot_y), 0],
                                        [0, 0, 0, 1]])


                corners = np.dot(corners, rot_y_matrix)

                
                #get translation
                tx = float(line[11])
                ty = float(line[12])
                tz = float(line[13])

                #add translation to corners
                corners[:, 0] += tx
                corners[:, 1] += ty
                corners[:, 2] += tz


                #make rectification homogenous

                #project corners from camera to velodyne
  
                corners = np.dot(np.linalg.inv(R0_rect), corners.T).T
                corners = np.dot(corners, inverse_rigid_trans(Tr_velo_to_cam).T)

                #remove homogenous
                corners = corners[:, :3]
                car_corners = np.vstack((car_corners, corners)) if car_corners.size else corners


                #update the bounding box from corners
                x_min = np.min(corners[:, 0])
                x_max = np.max(corners[:, 0])
                y_min = np.min(corners[:, 1])
                y_max = np.max(corners[:, 1])
                z_min = np.min(corners[:, 2])
                z_max = np.max(corners[:, 2])

                #get points in the bounding box
                points_in_box = pointcloud[(pointcloud[:, 0] > x_min) & (pointcloud[:, 0] < x_max) & (pointcloud[:, 1] > y_min) & (pointcloud[:, 1] < y_max) & (pointcloud[:, 2] > z_min) & (pointcloud[:, 2] < z_max)]

                #if less than 8 points in the bounding box, remove the label

                #distance to the center of the bounding box
                #distance from 0 to the center of the bounding box
                dist = np.sqrt((tx ** 2) + (ty ** 2) + (tz ** 2))

                if points_in_box.shape[0] < bbc:
                    lines[lc] = ''
                    print('removed label', label_file, line[0], points_in_box.shape[0], dist)

                #if the distance is more than 80 meters, remove the label
                elif dist > 85:
                    lines[lc] = ''
                    print('removed label because of distance', label_file, line[0], points_in_box.shape[0], dist)



            else:
                print(label_file, line[0])
            lc += 1

        #write the new label file
        with open(label_out_path + label_file, 'w') as f:
            for line in lines:
                f.write("%s" % line)

    #print numb objects before
    label_files = [f for f in os.listdir(label_path) if f.endswith('.txt')]
    num_objects_before = 0
    for label_file in label_files:
        with open(label_path + label_file, 'r') as f:
            lines = f.readlines()
            num_objects_before += len(lines)


    #print numb objects after
    label_files = [f for f in os.listdir(label_out_path) if f.endswith('.txt')]
    num_objects = 0


    #if training mode, delete train.txt and val.txt and create new ones
    if mode == 'training':
        if os.path.exists(set_path_out + 'train.txt'):
            os.remove(set_path_out + 'train.txt')
        if os.path.exists(set_path_out + 'val.txt'):
            os.remove(set_path_out + 'val.txt')

    elif mode == 'testing':
        if os.path.exists(set_path_out + 'test.txt'):
            os.remove(set_path_out + 'test.txt')


    for label_file in label_files:
        with open(label_out_path + label_file, 'r') as f:
            lines = f.readlines()
            num_objects += len(lines)
            if len(lines) > 0:
                #if there are still labels, add to train.txt and val.txt in /ImageSets/
                #if path is training, add to train.txt and val.txt
                #if path is testing,  add to test.txt
                if mode == 'training':
                    #if name is in train.txt, add to train.txt otherwise add to val.txt look up in /ImageSetsRaw/

                    with open(set_path + 'train.txt', 'r') as f:
                        train_lines = f.read().splitlines()
                        train_lines = [line.rstrip('\n') for line in train_lines] # remove newline character
                        if label_file[:-4] in train_lines:
                            with open(set_path_out + 'train.txt', 'a') as f:
                                f.write("%s\n" % label_file[:-4])

                    with open(set_path + 'val.txt', 'r') as f:
                        val_lines = f.read().splitlines()
                        if label_file[:-4] in val_lines:
                            with open(set_path_out + 'val.txt', 'a') as f:
                                f.write("%s\n" % label_file[:-4])

                elif mode == 'testing':
                    with open(set_path + 'test.txt', 'r') as f:
                        test_lines = f.read().splitlines()
                        if label_file[:-4] in test_lines:
                            with open(set_path_out + 'test.txt', 'a') as f:
                                f.write("%s\n" % label_file[:-4])

       

        

    print('num objects before', num_objects_before, 'num objects after', num_objects)



def zero_inten (file_path):
    files = [f for f in os.listdir(file_path) if f.endswith('.bin')]
    for file in files:
        pointcloud = np.fromfile(file_path + file, dtype=np.float32, count=-1).reshape([-1, 4])
        pointcloud[:, 3] = 0
        pointcloud.tofile(file_path + file)


from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
#main function
if __name__ == '__main__':



    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('--bbc', type=int, default=0, help='bounding box cutoff')
 
    args = parser.parse_args()

    path = '/home/oq55olys/Projects/detection/OpenPCDet/data/kitti/training/'

    zero_inten(path + 'velodyne/')

    remove_empty_labels(path, args.bbc)

    #path=  '/home/oq55olys/Projects/detection/OpenPCDet/data/kitti/testing/'

    #remove_empty_labels(path, args.bbc)