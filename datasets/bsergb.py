from .common import RawVFIData


class BSERGB_Train(RawVFIData):
    def __init__(self, data_root, **kwargs):
        super().__init__(data_root, dataset='bsergb', training=True, **kwargs)


class BSERGB_Test(RawVFIData):
    def __init__(self, data_root, skips=(1,3), **kwargs):
        super().__init__(data_root, dataset='bsergb', skips=skips, training=False, num_targets=None, **kwargs)

