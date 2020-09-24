import cv2
import numpy as np
import torch
import json

from NTURGBD_Datasets import Skeleton_Datasets

def simple_draw(s_data):
    totaledges = [[1,2],[1,13],[1,17],
          [2,21],[3,4],[3,21],
          [5,6],[5,21],[6,7],
          [7,8],[8,22],[8,23],
          [9,21],[9,10],[10,11],
          [11,12],[12,24],[12,25],
          [13,14],[14,15],[15,16],
          [17,18],[18,19],[19,20]]
    times = s_data.size(1)
    for t in range(times):
        canvas = np.zeros([1080,1920,3],dtype=np.uint8)
        tmp_data = s_data[:,t,:,0].transpose(1,0) # nxc
        points = []
        for joint_id in range(tmp_data.size(0)):
            x = tmp_data[joint_id][0].item()+1
            y = tmp_data[joint_id][1].item()+1
            x = int(x*960)
            y = int(y*540)

            points.append([x,y])
            cv2.circle(canvas, (x,y), 5, (255,255,255),-1)
            
        for ed in totaledges:
            #print(ed)
            p1 = points[ed[0]-1]
            p2 = points[ed[1]-1]
            cv2.line(canvas, (p1[0],p1[1]), (p2[0],p2[1]), (0, 0, 255), 3)

        cv2.namedWindow("frame", 0)
        cv2.imshow("frame", canvas)
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()



if __name__ == "__main__":

    train_dset = Skeleton_Datasets(root_path="./nturgb+d_skeletons/", 
                                   istrain=True, 
                                   mode="CrossView", 
                                   max_human_in_clips=2, 
                                   fast_frames_num=32, 
                                   slow_rate=8, 
                                   fast_rate=1) 

    loader_params = {"batch_size":16, "shuffle":True, "num_workers":4}

    train_loader = torch.utils.data.DataLoader(train_dset, **loader_params)

    for data_dict in train_loader:
        print(type(data_dict))
        for name in data_dict:
            print(name)
            data_under_label = data_dict[name]
            if type(data_under_label) == type({}):
                for name2 in data_under_label:
                    print("   ", name2, " shape:  ", data_under_label[name2].size())

        print()
        print()
        simple_draw(s_data=data_dict["fast"]["data"][0])
        print()
        print("adj self")
        print(data_dict["slow"]["adj_ori"][0,0,:10,:10,0])
        print()
        print("adj in")
        print(data_dict["slow"]["adj_in"][0,0,:10,:10,0])
        print()
        print("adj out")
        print(data_dict["slow"]["adj_out"][0,0,:10,:10,0])
        print()
        print("adj ori")
        print(train_dset.AdjM_ori[:10,:10])
        break


