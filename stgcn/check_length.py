import os

root_path = "/data/chou/Desktop/nturgb+d_skeletons/"
files = os.listdir(root_path)


threshold = 32

total_number = []
under_thre = []
total, sum_num = 0, 0
max_num, min_num = None, None
for file_id, file in enumerate(files):
    if file_id%500 == 0:
        print("{}/{}".format(file_id, len(files)))
    skeleton_path = root_path + "/" + file
    with open(skeleton_path, "r") as fske:
        frame_number = int(fske.read().split("\n")[0])

    if max_num is None or frame_number>max_num:
        max_num = frame_number
    if min_num is None or frame_number<min_num:
        min_num = frame_number

    sum_num += frame_number
    total += 1

    total_number.append(frame_number)

    if frame_number<threshold:
        under_thre.append({"path":root_path+"/"+file, "length":frame_number})

print("total video number: ", total)
print("average length: ", sum_num/total)
print("max length: ", max_num)
print("min length: ", min_num)
print("under threshold{} video number: {}".format(threshold, len(under_thre)))

for video_data in under_thre:
    vpath = video_data["path"]
    os.remove(vpath)
    print("del {}".format(vpath))

final_videos = os.listdir(root_path)

print("{} ---> {}".format(total, len(final_videos)))
