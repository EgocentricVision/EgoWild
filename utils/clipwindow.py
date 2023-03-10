import torch


class ClipWindow(object):
    def __init__(self, args, split):
        self.args = args
        self.split = split
        self.input = None
        self.modality = args.modality
        self.reset_input = True

    def get(self, data, i_c):
        if not (self.args.models.model == "Movinet" and self.args.models.causal and
                self.args[self.split].num_frames_per_clip[self.modality] == 1):
            if self.args[self.split].full_length:
                if self.args[self.split].detached_clips:
                    start_frame = i_c * self.args[self.split].num_frames_per_clip[self.modality]
                    stop_frame = (i_c + 1) * self.args[self.split].num_frames_per_clip[self.modality]
                    clip = data[:, start_frame:stop_frame, :, :, :].permute(0, 2, 1, 3, 4)
                else:
                    clip = data[:, i_c, :, :, :].unsqueeze(1)
                    if self.input is None:
                        clip = clip.repeat(1, self.args[self.split].num_frames_per_clip[self.modality], 1, 1, 1).permute(0, 2, 1, 3, 4)
                    else:
                        clip = clip.permute(0, 2, 1, 3, 4)
                        clip = torch.cat([self.input[:, :, 1:, :, :], clip], dim=2)
            else:
                clip = data[i_c]

            if self.args[self.split].full_length:
                self.input = clip
        else:
            return data[i_c]
        return clip
