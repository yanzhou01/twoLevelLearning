import pickle
import pandas as pd
import numpy as np
import re

file_name = '/Volumes/GoogleDrive/我的云端硬盘/fashion_research/codes/features.pickle'
with open(file_name, 'rb') as handle:
    tempDictFeatures = pickle.load(handle)

data = pd.DataFrame.from_dict(tempDictFeatures, orient = 'index')

tags_raw = data.coordinate

temp_str =  'カーディガン (ブラウン系)\nパンツ (ベージュ系)\n腕時計 (ベージュ系)\nイヤリング（両耳用） (ベージュ系)\nスニーカー (ブラック系)\n'

def cleanse_tag(tag):
    tag = tag.split("\n")
    del tag[-1] # remove last void element
    # split str and (str)
    feature = []
    for element in tag:
        try:
            feature.extend(element_split(element))
        except:
            continue
    return feature

def element_split(element):
    cat = element[:element.index(" (")]
    col = element[element.index(" (")+2:element.index("系")]
    return cat, col


## test

feature1 = cleanse_tag(tags_raw[1])
print(feature1)

tags_clean = {}
for i, tag in enumerate(tags_raw):
    try:
        tags_clean[i] = cleanse_tag(tag)
    except:
        continue

## dict of category
path = "/Volumes/GoogleDrive/我的云端硬盘/fashion_research/codes/categories"
categories = pd.Series.from_csv(path)
colors = [
    "ホワイト", "ブラック", "グレー", "ブラウン", "ベージュ", "グリーン", "ブルー", "パープル", "イエロー", "ピンク", "レッド", "オレンジ", "シルバー", "ゴールド", "その他" 
]

columns = categories.index.values.tolist()
color_col = []
for column in columns:
    col_to_append = column+"color"
    print(col_to_append)
    color_col.append(col_to_append)
columns.extend(color_col)

index = [*tags_clean]

labels = pd.DataFrame("", index, columns)

#categories should be list instead of string!
#run from get_cats.py to get categrories

for key, tag in tags_clean.items():
    print(key)
    for j in range(len(tag)):
        to_match = tag[j]
        for p, cat in enumerate(categories):
            if to_match in cat:
                #print(p, cat)
                matched = categories.index[p]
                #print(matched)
                labels.loc[key, matched] = to_match
                col_to_append = matched+"color"
                labels.loc[key, col_to_append] = tag[j+1]

path2 = "/Volumes/GoogleDrive/我的云端硬盘/fashion_research/codes/labels.pkl"
labels.to_pickle(path2)

