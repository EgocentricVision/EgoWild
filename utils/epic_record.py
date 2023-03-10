from .video_record import VideoRecord


class EpicVideoRecord(VideoRecord):
    def __init__(self, tup, dataset_conf):
        self._index = str(tup[0])
        self._series = tup[1]
        self.dataset_conf = dataset_conf

    @property
    def uid(self):
        return self._series['uid']

    @property
    def untrimmed_video_name(self):
        return self._series['video_id']

    @property
    def kitchen(self):
        return int(self._series['video_id'].split('_')[0][1:])

    @property
    def kitchen_p(self):
        return self._series['video_id'].split('_')[0]

    @property
    def recording(self):
        return int(self._series['video_id'].split('_')[1])

    def start_frame(self, untrimmed=False):
        if untrimmed:
            return self._series['start_frame']
        return max(self._series['start_frame'] - 1, 0)

    @property
    def end_frame(self):
        return self._series['stop_frame'] - 2

    @property
    def num_frames(self):
        return {'RGB': self.end_frame - self.start_frame(),
                'Flow': int((self.end_frame - self.start_frame()) / 2),
                'Event': int((self.end_frame - self.start_frame()) / self.dataset_conf["Event"].rgb4e),
                'Spec': self.end_frame - self.start_frame()}

    @property
    def label(self):
        if 'verb_class' not in self._series.keys().tolist():
            raise NotImplementedError
        return self._series['verb_class']
