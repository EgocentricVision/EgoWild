import torch


class InfiniteBatchSampler(torch.utils.data.sampler.BatchSampler):
    """Sample batches from the dataset indefinitely"""

    def __init__(self, dataset, batch_size, drop_last):
        # create a random sampler to iterate over the dataset
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
        super(InfiniteBatchSampler, self).__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        while True:
            # sample batches from the dataset
            yield from super().__iter__()
            # once the entire dataset has been processed, start over

    def len(self):
        return None
