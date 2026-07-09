from torch.utils.data import ConcatDataset

from src.genimage_dataset import GenImageDataset


class MixedGenImageDataset(ConcatDataset):
    """
    Combines multiple GenImage generators into one dataset.
    """

    def __init__(self, root_dir, generators, split="train", transform=None):
        datasets = [
            GenImageDataset(
                root_dir=root_dir,
                generator=generator,
                split=split,
                transform=transform
            )
            for generator in generators
        ]

        super().__init__(datasets)