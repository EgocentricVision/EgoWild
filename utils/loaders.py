import glob
import math
from abc import ABC
import pandas as pd
from .epic_record import EpicVideoRecord
import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
from utils.logger import logger
import torch

''' 
CLASSES USED FOR THIS SETTING
        Verbs:
        0 - take (get)
        1 - put-down (put/place)
        2 - open
        3 - close
        4 - wash (clean)
        5 - cut
        6 - stir (mix)
        7 - pour
        Domains:
        D1 - P08
        D2 - P01
        D3 - P22
'''


class EpicKitchensDataset(data.Dataset, ABC):
    def __init__(self, split, modality, mode, dataset_conf, num_frames_per_clip, num_clips, dense_sampling,
                 transform=None, additional_info=False, load_feat=False, untrimmed=False, full_length=False,
                 consecutive_clips=False, **kwargs):
        """
        split: str (D1, D2 or D3)
        modality: str
        mode: str (train, test/val)
        dataset_conf must contain the following:
            - annotations_path: str
            - stride: int
        dataset_conf[modality] for the modalities used must contain:
            - data_path: str
            - tmpl: str
            - (Event only) rgb4e: int
        num_frames_per_clip: dict(modality: int)
        num_clips: int
        dense_sampling: dict(modality: bool)
        additional_info: bool, set to True if you want to receive also the uid and the video name from the get function
        load_feat: bool, return features instead of samples
        untrimmed: bool, returns all untrimmed frames if set to True in validation/test
        consecutive_clips: bool, (train-only argument) set to True if you want the clips of the training split to be
            consecutive in training
        """
        self.modality = modality  # considered modality
        self.mode = mode  # 'train', 'val' or 'test'
        self.dataset_conf = dataset_conf
        self.num_frames_per_clip = num_frames_per_clip
        self.dense_sampling = dense_sampling
        self.num_clips = num_clips
        self.additional_info = additional_info
        self.split = split
        self.untrimmed = untrimmed
        self.consecutive_clips = consecutive_clips
        if self.untrimmed:
            untrimmed_string = "_untrimmed_multilabel"
        else:
            untrimmed_string = ""

        if self.mode == "train":
            pickle_name = split + "_train.pkl"
        elif kwargs.get('save', None) is not None:
            pickle_name = split + "_" + kwargs["save"] + ".pkl"
        else:
            pickle_name = split + "_test" + untrimmed_string + ".pkl"

        self.list_file = pd.read_pickle(os.path.join(self.dataset_conf.annotations_path, pickle_name))
        logger.info(f"Dataloader for {split}-{self.mode} with {len(self.list_file)} samples generated")

        self.video_list = [EpicVideoRecord(tup, self.dataset_conf) for tup in self.list_file.iterrows()]
        self.transform = transform  # pipeline of transforms
        self.full_length = full_length

        self.load_feat = load_feat
        if self.load_feat:
            self.model_features = None

            self.model_features = pd.DataFrame(pd.read_pickle(os.path.join("saved_features",
                                                                      self.dataset_conf[self.modality].features_name + "_" + self.modality + "_" +
                                                                      pickle_name))['features'])[["uid", "features_" + self.modality]]

            self.model_features['uid'] = self.model_features['uid'].astype(int)
            self.model_features = pd.merge(self.model_features, self.list_file, how="inner", on="uid")

    def _get_train_indices(self, record, modality='RGB'):
        stride = self.dataset_conf[modality].stride

        if self.dense_sampling[modality]:

            if self.consecutive_clips and self.num_clips > 1:
                collected_frames = self.num_clips * self.num_frames_per_clip[modality]
                idces_start = 0 if record.num_frames[modality] <= collected_frames * stride else \
                    np.random.randint(low=0, high=record.num_frames[modality] - (collected_frames * stride))
                indices = [x for x in range(idces_start, idces_start + collected_frames * stride, stride)]
            else:
                # selecting one frame and discarding another (alternation), to avoid duplicates
                center_frames = np.linspace(0, record.num_frames[modality], self.num_clips + 2,
                                            dtype=np.int32)[1:-1]

                indices = [x for center in center_frames for x in
                           range(center - math.ceil(self.num_frames_per_clip[modality] / 2 * stride),
                                 # start of the segment
                                 center + math.ceil(self.num_frames_per_clip[modality] / 2 * stride),
                                 # end of the segment
                                 stride)]  # step of the sampling

                offset = -indices[0] if indices[0] < 0 else 0
                for i in range(0, len(indices), self.num_frames_per_clip[modality]):
                    indices_old = indices[i]
                    for j in range(self.num_frames_per_clip[modality]):
                        indices[i + j] = indices[i + j] + offset if indices_old < 0 else indices[i + j]

            return indices

        else:
            indices = []
            # average_duration is the average stride among frames in the clip to obtain a uniform sampling BUT
            # the randint shifts a little (to add randomicity among clips)
            average_duration = record.num_frames[modality] // self.num_frames_per_clip[modality]
            if average_duration > 0:
                for _ in range(self.num_clips):
                    frame_idx = np.multiply(list(range(self.num_frames_per_clip[modality])), average_duration) + \
                                randint(average_duration, size=self.num_frames_per_clip[modality])
                    indices.extend(frame_idx.tolist())
            else:
                indices = np.zeros((self.num_frames_per_clip[modality] * self.num_clips,))
            indices = np.asarray(indices)

        return indices

    def _get_val_indices(self, record, modality):
        stride = self.dataset_conf[modality].stride

        max_frame_idx = max(1, record.num_frames[modality])
        if self.dense_sampling[modality]:

            if self.full_length:
                indices = [x - record.start_frame() for x in range(record.start_frame(), record.end_frame + 1, stride)]
                if len(indices) < self.num_frames_per_clip[modality]:
                    indices = np.concatenate([indices, (record.end_frame - record.start_frame() + 1)
                                              * np.ones(self.num_frames_per_clip[modality] - len(indices))])
                return indices

            n_clips = self.num_clips
            center_frames = np.linspace(0, record.num_frames[modality], n_clips + 2, dtype=np.int32)[1:-1]

            indices = [x for center in center_frames for x in
                       range(center - math.ceil(self.num_frames_per_clip[modality] / 2 * stride),
                             # start of the segment
                             center + math.ceil(self.num_frames_per_clip[modality] / 2 * stride),
                             # end of the segment
                             stride)]  # step of the sampling

            offset = -indices[0] if indices[0] < 0 else 0
            for i in range(0, len(indices), self.num_frames_per_clip[modality]):
                indices_old = indices[i]
                for j in range(self.num_frames_per_clip[modality]):
                    indices[i + j] = indices[i + j] + offset if indices_old < 0 else indices[i + j]

            return indices

        else:  # uniform sampling
            # Code for "Deep Analysis of CNN-based Spatio-temporal Representations for Action Recognition"
            # arXiv: 2104.09952v1
            # Yuan Zhi, Zhan Tong, Limin Wang, Gangshan Wu
            frame_idices = []
            sample_offsets = list(range(-self.num_clips // 2 + 1, self.num_clips // 2 + 1))
            for sample_offset in sample_offsets:
                if max_frame_idx > self.num_frames_per_clip[modality]:
                    tick = max_frame_idx / float(self.num_frames_per_clip[modality])
                    curr_sample_offset = sample_offset
                    if curr_sample_offset >= tick / 2.0:
                        curr_sample_offset = tick / 2.0 - 1e-4
                    elif curr_sample_offset < -tick / 2.0:
                        curr_sample_offset = -tick / 2.0
                    frame_idx = np.array([int(tick / 2.0 + curr_sample_offset + tick * x) for x
                                          in range(self.num_frames_per_clip[modality])])
                else:
                    np.random.seed(sample_offset - (-self.num_clips // 2 + 1))
                    frame_idx = np.random.choice(max_frame_idx, self.num_frames_per_clip[modality])
                frame_idx = np.sort(frame_idx)
                frame_idices.extend(frame_idx.tolist())
            frame_idx = np.asarray(frame_idices)
            return frame_idx

    def __getitem__(self, index):

        # record is a row of the pkl file containing one sample/action
        record = self.video_list[index]

        if self.load_feat:
            sample_row = self.model_features[self.model_features["uid"] == int(record.uid)]
            assert len(sample_row) == 1
            sample = sample_row["features_" + self.modality].values[0]
            if self.additional_info:
                return sample, record.label, record.untrimmed_video_name, record.uid
            else:
                return sample, record.label

        if self.mode == "train":
            segment_indices = self._get_train_indices(record, self.modality)
        else:
            if self.untrimmed:
                segment_indices = [0]
            else:
                segment_indices = self._get_val_indices(record, self.modality)

        img, label = self.get(self.modality, record, segment_indices)
        frames = img

        if self.untrimmed:
            return frames, {"label": label, "uid": record.uid}

        if self.additional_info:
            return frames, label, record.untrimmed_video_name, record.uid

        return frames, label

    def get(self, modality, record, indices):
        images = list()
        for frame_index in indices:
            p = int(frame_index)
            frame = self._load_data(modality, record, p)
            images.extend(frame)

        process_data = self.transform(images)
        return process_data, record.label

    def _load_data(self, modality, record, idx):
        data_path = self.dataset_conf[modality].data_path
        tmpl = self.dataset_conf[modality].tmpl

        if modality == 'RGB' or modality == 'RGBDiff':
            idx_untrimmed = record.start_frame(self.untrimmed) + idx
            try:
                img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(idx_untrimmed))) \
                    .convert('RGB')
            except (FileNotFoundError, OSError) as e:
                max_idx_video = int(sorted(glob.glob(os.path.join(data_path,
                                                                  record.untrimmed_video_name,
                                                                  "img_*")))[-1].split("_")[-1].split(".")[0])
                found = False
                skip = 0
                while not found:
                    idx_untrimmed += 1
                    skip += 1
                    if idx_untrimmed > max_idx_video:
                        img = Image.open(
                            os.path.join(data_path, record.untrimmed_video_name, tmpl.format(max_idx_video))) \
                            .convert('RGB')
                    else:
                        try:
                            img = Image.open(os.path.join(data_path, record.untrimmed_video_name,
                                                          tmpl.format(idx_untrimmed))).convert('RGB')
                            found = True
                            if skip > 1:
                                print("Img not found: " + str(skip) + " skipped frames")
                        except (FileNotFoundError, OSError) as e:
                            found = False
            return [img]
        elif modality == 'Flow':
            idx_untrimmed = (record.start_frame(self.untrimmed) // 2) + idx
            try:
                x_img = Image.open(os.path.join(data_path, record.untrimmed_video_name,
                                                tmpl.format('x', idx_untrimmed))).convert('L')
                y_img = Image.open(os.path.join(data_path, record.untrimmed_video_name,
                                                tmpl.format('y', idx_untrimmed))).convert('L')
            except FileNotFoundError:
                for i in range(0, 3):
                    found = True
                    try:
                        x_img = Image.open(os.path.join(data_path, record.untrimmed_video_name,
                                                        tmpl.format('x', idx_untrimmed + i))).convert('L')
                        y_img = Image.open(os.path.join(data_path, record.untrimmed_video_name,
                                                        tmpl.format('y', idx_untrimmed + i))).convert('L')
                    except FileNotFoundError:
                        found = False

                    if found:
                        break
            return [x_img, y_img]

        elif modality == 'Event':
            idx_untrimmed = (record.start_frame(self.untrimmed) // self.dataset_conf["Event"].rgb4e) + idx
            try:
                img_npy = np.load(os.path.join(data_path, record.untrimmed_video_name,
                                               tmpl.format(idx_untrimmed))).astype(np.float32)
            except FileNotFoundError:
                max_idx_video = int(sorted(glob.glob(os.path.join(self.dataset_conf['RGB'].data_path,
                                                                  record.untrimmed_video_name, "img_*")))[-1]
                                    .split("_")[-1].split(".")[0])
                if max_idx_video % 6 == 0:
                    max_idx_event = (max_idx_video // self.dataset_conf["Event"].rgb4e) - 1
                else:
                    max_idx_event = max_idx_video // self.dataset_conf["Event"].rgb4e
                if idx_untrimmed > max_idx_video:
                    img_npy = np.load(os.path.join(data_path, record.untrimmed_video_name,
                                                   tmpl.format(max_idx_event))).astype(np.float32)
                else:
                    raise FileNotFoundError
            return np.stack([img_npy], axis=0)
        else:
            raise NotImplementedError("Modality not implemented")

    def __len__(self):
        return len(self.video_list)
