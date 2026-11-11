from .common import RawVFIData


class GOPRO_Train(RawVFIData):
    def __init__(self, data_root, **kwargs):
        super().__init__(data_root, dataset='gopro', training=True, **kwargs)


class GOPRO_Test(RawVFIData):
    def __init__(self, data_root, skips=(7,15), **kwargs):
        super().__init__(data_root, dataset='gopro', skips=skips, training=False, num_targets=None, **kwargs)

