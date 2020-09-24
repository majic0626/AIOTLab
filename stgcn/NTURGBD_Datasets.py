"""
For speed, we modify the original dataloader code
In this program, we can

(1) store skeleton info into .json file (56880 files)
(2) read skeleton from .json file (56880 files)

ATTENTION PLEASE: we only store data_tensor in .json file without adjMatrix

"""

import torch
import numpy as np
import json
import os
import time


# this is train info , is index not in train info ==> test.
class Skeleton_Datasets(torch.utils.data.Dataset):

    def __init__(self,
                 root_path,
                 istrain,
                 mode,
                 max_human_in_clips,
                 fast_frames_num,
                 slow_rate,
                 fast_rate,
                 preprocess=False
                 ):
        """
        root_path: root path of all skeleton file. (.skeleton)(a folder contain text files).
        istrain: train or test (str).
        mode:CV or CrossView, CS or CrossSubject (str).

        """
        self.fast_frames_num = fast_frames_num
        self.slow_rate = slow_rate
        self.fast_rate = fast_rate
        self.root_path = root_path
        self.istrain = istrain
        self.preprocess = preprocess  # preprocess phase or train phase
        self.mode = mode
        # 3.2.2 Cross-View Evaluation
        if self.mode.lower() in ["cv", "crossview"]:
            self.train_index = [2, 3]
            prefix = "C"
        # 3.2.1 Cross-Subject Evaluation
        elif self.mode.lower() in ["cs", "crosssubject"]:
            self.train_index = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
            prefix = "S"
        # get train string
        for n, idx in enumerate(self.train_index):
            idx3 = str(idx).zfill(3)
            self.train_index[n] = prefix + idx3

        # print(self.train_index)

        self.joint_num = 25  # kinetic v2 output.
        self.edges = [[1, 2], [1, 13], [1, 17],
                      [2, 21], [3, 4], [3, 21],
                      [5, 6], [5, 21], [6, 7],
                      [7, 8], [8, 22], [8, 23],
                      [9, 21], [9, 10], [10, 11],
                      [11, 12], [12, 24], [12, 25],
                      [13, 14], [14, 15], [15, 16],
                      [17, 18], [18, 19], [19, 20]]
        # we only use: 'x', 'y', 'z', 'depthX', 'depthY', 'orientationW', 'orientationX', 'orientationY','orientationZ'
        self.joint_info_key = ['x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                               'orientationW', 'orientationX', 'orientationY',
                               'orientationZ', 'trackingState']
        # we don't use any body info
        self.body_info_key = ['bodyID', 'clipedEdges', 'handLeftConfidence',
                              'handLeftState', 'handRightConfidence', 'handRightState',
                              'isResticted', 'leanX', 'leanY', 'trackingState']

        self.connects_dict, self.AdjM_ori = self.getConnectionandAdjancyMatrix()

        self.datas_info = self.getDatasInfo()

        self.max_human_in_clips = max_human_in_clips
        self.json_root = '/data/chou/Desktop/nturgbd/nturgbd_skeleton_json'

    def getDatasInfo(self):
        """
        make a data info list.
        for __getitem__ and __len__
        """
        datas_info = []
        skeleton_files = os.listdir(self.root_path)
        for sfile in skeleton_files:
            if not self.preprocess:
                for cheching_str in self.train_index:
                    if cheching_str in sfile:
                        if self.istrain:
                            skip = False
                        else:
                            skip = True
                        break
                    else:
                        if self.istrain:
                            skip = True
                        else:
                            skip = False
                if skip:
                    continue
            class_int = int(sfile.split(".")[0].split("A")[-1])
            file_path = self.root_path + "/" + sfile
            tmp_info = {"path": file_path, "label": class_int - 1}
            datas_info.append(tmp_info)
        return datas_info

    def __len__(self):
        return len(self.datas_info)

    def __getitem__(self, index):
        """
        return example:
            fast
                data  shape:   torch.Size([16, 9, 32, 25, 2]) # B x C x T x N x M
                m_number  shape:   torch.Size([16, 32, 1]) # B x T x 1
                adj_ori  shape:   torch.Size([16, 32, 25, 25, 2]) # B x T x N x N x M
                adj_in  shape:   torch.Size([16, 32, 25, 25, 2]) # B x T x N x N x M
                adj_out  shape:   torch.Size([16, 32, 25, 25, 2]) # B x T x N x N x M
            slow
                data  shape:   torch.Size([16, 9, 4, 25, 2])
                m_number  shape:   torch.Size([16, 4, 1])
                adj_ori  shape:   torch.Size([16, 4, 25, 25, 2])
                adj_in  shape:   torch.Size([16, 4, 25, 25, 2])
                adj_out  shape:   torch.Size([16, 4, 25, 25, 2])
            label

        """
        file_info = self.datas_info[index]
        file_path = file_info["path"]

        if not self.preprocess:  # train or test phase
            with open("{}".format(self.json_root + '/' + file_path.split('/')[-1].split('.')[0] + '.json'), 'r') as f:
                sk = json.load(f)
            data_tensor, label_int = torch.from_numpy(np.array(sk["data"], dtype=np.float32)), int(sk["label"])
            final_datas = self.select_data(data_tensor, label_int, file_info)
            return final_datas

        else:  # store json phase
            data_tensor, label_int = self.load_entir_skeleton(file_info)
            data_json = {"data": data_tensor.numpy().tolist(),
                         "label": label_int}
            with open("{}".format(self.json_root + '/' + file_path.split('/')[-1].split('.')[0] + '.json'), 'w') as f:
                json.dump(data_json, f)
                print("save json file in {}".format(self.json_root + '/' + file_path.split('/')[-1].split('.')[0] + '.json'))
            return {"fast": {"data": torch.zeros(1, dtype=torch.float32)}, "slow": {"data": torch.zeros(1, dtype=torch.float32)}, "label": -1}

    def getConnectionandAdjancyMatrix(self):
        connectsM = {}  # store every joints connections.
        adjM = torch.zeros([self.joint_num, self.joint_num], dtype=torch.float32)
        for edge in self.edges:
            joint1, joint2 = edge
            joint1, joint2 = joint1 - 1, joint2 - 1
            if str(joint1) not in connectsM:
                connectsM[str(joint1)] = []
            if str(joint2) not in connectsM:
                connectsM[str(joint2)] = []

            connectsM[str(joint1)].append(joint2)
            connectsM[str(joint2)].append(joint1)

            adjM[joint1][joint2] = 1
            adjM[joint2][joint1] = 1

        adjM += torch.eye(self.joint_num, dtype=torch.float32)

        return connectsM, adjM

    def load_entir_skeleton(self, file_info):

        file_path = file_info["path"]
        contents = None
        with open(file_path, "r") as ftxt:
            contents = ftxt.read().split("\n")

        # first line
        clip_frames_num = int(contents[0].replace(" ", ""))

        # if change input features, change size of this tensor
        data_tensor = torch.zeros([clip_frames_num, self.max_human_in_clips, 25, 9], dtype=torch.float32)
        # M_number = torch.zeros([clip_frames_num, 1], dtype=torch.int32)
        # adj_ori = torch.zeros([clip_frames_num, self.max_human_in_clips, self.joint_num, self.joint_num], dtype=torch.float32)
        # adj_in = torch.zeros([clip_frames_num, self.max_human_in_clips, self.joint_num, self.joint_num], dtype=torch.float32)
        # adj_out = torch.zeros([clip_frames_num, self.max_human_in_clips, self.joint_num, self.joint_num], dtype=torch.float32)

        frame_num = -1
        human_counter, skeleton_id = 0, 0
        skeleton_line_counter = 0
        for line_id, line_content in enumerate(contents[1:]):

            sd = line_content.split(" ")

            if len(sd) == 1:
                if sd[0] == "25":
                    skeleton_id = 0
                    human_counter += 1
                else:
                    # not enough human in a clip
                    if human_counter < self.max_human_in_clips - 1:
                        remiander = self.max_human_in_clips - 1 - human_counter  # number
                        if remiander < 2:
                            data_tensor[frame_num][human_counter + 1] = data_tensor[frame_num][human_counter]
                            # adj_ori[frame_num][human_counter + 1] = adj_ori[frame_num][human_counter]
                            # adj_in[frame_num][human_counter + 1] = adj_in[frame_num][human_counter]
                            # adj_out[frame_num][human_counter + 1] = adj_out[frame_num][human_counter]
                        else:
                            data_tensor[frame_num][human_counter + 1:] = data_tensor[frame_num][human_counter].repeat(remiander, 1, 1)
                            # adj_ori[frame_num][human_counter + 1:] = adj_ori[frame_num][human_counter].repeat(remiander, 1, 1)
                            # adj_in[frame_num][human_counter + 1:] = adj_in[frame_num][human_counter].repeat(remiander, 1, 1)
                            # adj_out[frame_num][human_counter + 1:] = adj_out[frame_num][human_counter].repeat(remiander, 1, 1)

                    human_counter = -1
                    frame_num += 1  # next frame

            if len(sd) != 12:
                continue

            x, y, z, depthX, depthY, orientW, orientX, orientY, orientZ = \
               [float(x) for x in sd[:5]] + [float(x) for x in sd[7:11]]

            if human_counter > self.max_human_in_clips - 1:
                continue

            # assign value to datas
            data_tensor[frame_num][human_counter][skeleton_id][0] = x
            data_tensor[frame_num][human_counter][skeleton_id][1] = y
            data_tensor[frame_num][human_counter][skeleton_id][2] = z
            data_tensor[frame_num][human_counter][skeleton_id][3] = depthX
            data_tensor[frame_num][human_counter][skeleton_id][4] = depthY
            data_tensor[frame_num][human_counter][skeleton_id][5] = orientW
            data_tensor[frame_num][human_counter][skeleton_id][6] = orientX
            data_tensor[frame_num][human_counter][skeleton_id][7] = orientY
            data_tensor[frame_num][human_counter][skeleton_id][8] = orientZ

            skeleton_id += 1  # next skeleton(joint)

            # if skeleton_id == self.joint_num - 1:
            #     tmp_adj_ori, tmp_adj_in, tmp_adj_out = self.SpatialConfigPart_Dynamic(data_f=data_tensor[frame_num, human_counter, :, :3])
            #     adj_in[frame_num][human_counter] = tmp_adj_in
            #     adj_out[frame_num][human_counter] = tmp_adj_out
            #     adj_ori[frame_num][human_counter] = tmp_adj_ori

        label_int = file_info["label"]
        # return data_tensor, M_number, adj_ori, adj_in, adj_out, label_int
        return data_tensor, label_int  # will store this to file

    def SpatialConfigPart_Dynamic(self, data_f):
        """
        data_f: 25x3 tensor (x, y, z)
        """
        adj_ori = torch.eye(self.joint_num, dtype=torch.float32)
        adj_in = torch.zeros([self.joint_num, self.joint_num], dtype=torch.float32)
        adj_out = torch.zeros([self.joint_num, self.joint_num], dtype=torch.float32)
        mean_p = torch.mean(data_f, dim=0).repeat(self.joint_num, 1)  # 重心
        diff = data_f - mean_p
        total_distances = torch.sum(diff * diff, dim=1)  # size:25
        for joint_id in self.connects_dict:
            root_id = int(joint_id)
            conn_ids = self.connects_dict[joint_id]  # connections list
            root_dis = total_distances[root_id]
            for cid in conn_ids:
                tmp_dis = total_distances[cid]
                if tmp_dis > root_dis + 1e-5:
                    # 離心
                    adj_in[root_id][cid] += 1
                elif tmp_dis < root_dis - 1e-5:
                    # 向心
                    adj_out[root_id][cid] += 1
                else:
                    # 同等
                    adj_ori[root_id][cid] += 1

        adj_in += torch.eye(self.joint_num, dtype=torch.float32)
        adj_out += torch.eye(self.joint_num, dtype=torch.float32)

        return adj_ori, adj_in, adj_out

    # def select_data(self, data_tensor, M_number, adj_ori, adj_in, adj_out, label, file_info):
    def select_data(self, data_tensor, label, file_info):
        alpha = self.slow_rate / self.fast_rate
        slow_frames_num = int(self.fast_frames_num / alpha)

        # create data space.
        slow_data = torch.zeros([slow_frames_num, self.max_human_in_clips, 25, 9], dtype=torch.float32)
        fast_data = torch.zeros([self.fast_frames_num, self.max_human_in_clips, 25, 9], dtype=torch.float32)
        # slow_M_number = torch.zeros([slow_frames_num, 1], dtype=torch.int32)
        # fast_M_number = torch.zeros([self.fast_frames_num, 1], dtype=torch.int32)
        # slow_adj_ori = torch.zeros([slow_frames_num, self.max_human_in_clips, self.joint_num, self.joint_num], dtype=torch.float32)
        # fast_adj_ori = torch.zeros([self.fast_frames_num, self.max_human_in_clips, self.joint_num, self.joint_num], dtype=torch.float32)
        # slow_adj_in = torch.zeros_like(slow_adj_ori)
        # fast_adj_in = torch.zeros_like(fast_adj_ori)
        # slow_adj_out = torch.zeros_like(slow_adj_ori)
        # fast_adj_out = torch.zeros_like(fast_adj_ori)

        frames_needed = int(self.fast_rate * self.fast_frames_num)
        total_frame_number = data_tensor.size(0)
        choose_space = total_frame_number - frames_needed
        if choose_space < 0:
            choose_start_indexes = 0
        else:
            choose_start_indexes = torch.randint(0, choose_space + 1, (1,)).item()
        fast_counter, slow_counter = 0, 0
        while True:
            if fast_counter == self.fast_frames_num and slow_counter == slow_frames_num:
                break

            if fast_counter < self.fast_frames_num:
                rdata_index = int(choose_start_indexes + fast_counter * self.fast_rate)

                offset_factor = int(rdata_index / total_frame_number)
                rdata_index -= offset_factor * total_frame_number

                fast_data[fast_counter] = data_tensor[rdata_index]
                # fast_M_number[fast_counter] = M_number[rdata_index]
                # fast_adj_ori[fast_counter] = adj_ori[rdata_index]
                # fast_adj_in[fast_counter] = adj_in[rdata_index]
                # fast_adj_out[fast_counter] = adj_out[rdata_index]
                fast_counter += 1

            if slow_counter < slow_frames_num:
                rdata_index = int(choose_start_indexes + slow_counter * self.slow_rate)

                offset_factor = int(rdata_index / total_frame_number)
                rdata_index -= offset_factor * total_frame_number

                slow_data[slow_counter] = data_tensor[rdata_index]
                # slow_M_number[slow_counter] = M_number[rdata_index]
                # slow_adj_ori[slow_counter] = adj_ori[rdata_index]
                # slow_adj_in[slow_counter] = adj_in[rdata_index]
                # slow_adj_out[slow_counter] = adj_out[rdata_index]
                slow_counter += 1

        fianl_datas = {"fast": {"data": fast_data.permute(3, 0, 2, 1),
                                # "m_number": fast_M_number,
                                # "adj_ori": fast_adj_ori.permute(0, 2, 3, 1),
                                # "adj_in": fast_adj_in.permute(0, 2, 3, 1),
                                # "adj_out": fast_adj_out.permute(0, 2, 3, 1)
                                },
                       "slow": {"data": slow_data.permute(3, 0, 2, 1),
                                # "m_number": slow_M_number,
                                # "adj_ori": slow_adj_ori.permute(0, 2, 3, 1),
                                # "adj_in": slow_adj_in.permute(0, 2, 3, 1),
                                # "adj_out": slow_adj_out.permute(0, 2, 3, 1)
                                },
                       "label": label}

        return fianl_datas


if __name__ == "__main__":
    bs = 256  # batch size
    m = 2  # man
    fn = 64  # frame number
    alpha = 8  # alpha
    fast_rate = 2  # rate for fast stream
    slow_rate = fast_rate * alpha  # rate for slow stream
    c = 3
    data2json = True  # false = (train/test) or true = (prepare json)
    trainset = Skeleton_Datasets(root_path='/data/chou/Desktop/nturgbd/nturgbd_skeleton',
                                 istrain=False,
                                 mode='cv',
                                 max_human_in_clips=m,
                                 fast_frames_num=fn,
                                 slow_rate=slow_rate,
                                 fast_rate=fast_rate,
                                 preprocess=data2json
                                 )
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=bs,
                                              shuffle=False,
                                              num_workers=16)

    s = time.time()
    for ix, sample in enumerate(trainloader):
        data_fast, data_slow, label = sample["fast"]["data"], sample["slow"]["data"], sample["label"]
    e = time.time()
    print(data_fast.size(), data_slow.size(), label)
    print("time cost for prepaeing data: ", e - s)
