#!/usr/bin/env python
# coding: utf-8


import pandas as pd

multi_label = True
for split in ["D1", "D2", "D3"]:
    df = pd.read_pickle(split+"_test.pkl")
    if split not in ["D1", "D2", "D3"]:
        df["uid"], _ = pd.factorize(df.index)
    set_video = set(df["video_id"])
    last_frame = {v: max(df[df["video_id"] == v]["stop_frame"]) for v in set_video}
    frames = {"video_id": [], "start_frame": [], "verb_class": [], "last_frame": [], "uid": []}
    for v in set_video:
        for i in range(0, last_frame[v]+1):
            labels = df[(df["video_id"] == v) & (df["start_frame"] <= i) & (df["stop_frame"] >= i)]
            label_dict = labels[labels.stop_frame == labels.stop_frame.min()].head(1)
            if label_dict.empty:
                if multi_label:
                    label = [-1]
                    uid = [-1]
                else:
                    label = -1
                    uid = -1
            else:
                if multi_label:
                    label = list(int(lab["verb_class"]) for _, lab in labels.iterrows())
                    uid = list(int(lab["uid"]) for _, lab in labels.iterrows())
                else:
                    label = int(label_dict["verb_class"])
                    uid = int(label_dict["uid"])
            frames["video_id"].append(v)
            frames["start_frame"].append(i)
            frames["verb_class"].append(label)
            if not multi_label:
                frames["last_frame"].append(i == label_dict.stop_frame.min())
            else:
                frames["last_frame"].append(False)
            frames["uid"].append(uid)

    pd.DataFrame.from_dict(frames).to_pickle(split+"_test_untrimmed_multilabel.pkl", protocol=4)
