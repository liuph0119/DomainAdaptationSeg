import os

data_dir = "F:/ADDA_Seg/data/inria_test/target_image"
save_file= "F:/ADDA_Seg/data/inria_test/target.txt"

with open(save_file, "w", encoding="utf-8") as f:
    for fn in os.listdir(data_dir):
        f.write("%s\n" % fn.split(".")[0])