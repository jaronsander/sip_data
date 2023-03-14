import os
import shutil
import pandas as pd

dir = 'refined_window'

def bucket_files(dir_list, rootdir, num_buckets):
    df = []
    for dir in dir_list:
        for root,sf,files in os.walk(dir):
            print(root,sf,files)
            for f in files:
                index = int(f.split('_')[1])
                label = int(f.split('_')[2].split('.')[0])
                fold = index % num_buckets
                subfold = os.path.join(rootdir,"fold"+str(fold))
                print(f,subfold,label)
                if not os.path.isdir(subfold):
                    os.makedirs(subfold)
                shutil.move(os.path.join(root, f), subfold)
                df.append([f,subfold,label])
    annotations = pd.DataFrame(df,columns=["file","folder","label"])
    annotations.to_csv("fall_annotations.csv")


def annotate_files(dir_list):
    df = []
    for dir in dir_list:
        for root,sf,files in os.walk(dir):
            print(root,sf,files)
            for f in files:
                label = int(f.split('_')[2].split('.')[0])
                df.append([f, root, label])
    annotations = pd.DataFrame(df, columns=["file", "folder", "label"])
    annotations.to_csv("fall_annotations.csv")

if __name__ == "__main__":
    # bucket_files(['poopdir1','poopdir2'],'',2)
    annotate_files(['ambient','falls3'])