from dataclasses import asdict, dataclass, fields


@dataclass
class ModelConfig:
    voxel_bins: int = 128
    nb_of_flow: int = 16
    refine: str = 'transformer'
    unit_dim: int = 32
    legacy_query: bool = False
    legacy_warp: bool = False
    mask_in_refine: bool = False
    basr: bool = False
    basr_heads: tuple = (4, 2)
    basr_chunk_size: int = 256
    basr_support_ratio: float = 1.0
    basr_keys: str = 'warped'
    flow_corr: bool = False
    correlation_radius: int = 4
    splat_metric: bool = False
    splat_backend: str = 'auto'
    w_flow: float = 0.0
    w_ps: float = 0.0
    checkpoint_bifeb: bool = False
    checkpoint_refine: bool = False

    @classmethod
    def from_args(cls, args=None):
        if isinstance(args, cls):
            result = cls(**asdict(args))
        else:
            source = args if isinstance(args, dict) else vars(args) if args is not None else {}
            result = cls(**{f.name: source[f.name] for f in fields(cls) if f.name in source})
        result.basr_heads = tuple(result.basr_heads)
        result.validate()
        return result

    def validate(self):
        if self.voxel_bins < 2 or self.nb_of_flow <= 0 or self.voxel_bins % self.nb_of_flow:
            raise ValueError('voxel_bins must be >=2 and divisible by positive nb_of_flow')
        if self.refine not in ('resblock', 'transformer') or self.unit_dim not in (32, 44):
            raise ValueError('refine must be resblock/transformer; unit_dim must be 32/44')
        if self.basr_keys != 'warped':
            raise ValueError('raw BASR keys need source-grid mask mapping and are not enabled')
        if len(self.basr_heads) != 2 or any(h <= 0 for h in self.basr_heads):
            raise ValueError('basr_heads requires two positive integers')
        if self.unit_dim % self.basr_heads[0] or (self.unit_dim // 2) % self.basr_heads[1]:
            raise ValueError('BASR embedding dimensions must be divisible by the head counts')
        if self.basr_chunk_size <= 0 or not 0 < self.basr_support_ratio <= 1:
            raise ValueError('invalid BASR chunk size or support ratio')
        if self.correlation_radius < 0 or self.w_flow < 0 or self.w_ps < 0:
            raise ValueError('radii and loss weights must be nonnegative')
        if (self.basr or self.mask_in_refine) and self.refine != 'transformer':
            raise ValueError('BASR and mask_in_refine require transformer refinement')
        if self.w_ps > 0 and not self.basr:
            raise ValueError('w_ps > 0 requires BASR')
        if self.splat_metric and self.w_flow <= 0:
            raise ValueError('splat_metric requires differentiable forward photometric loss (w_flow>0)')
        if self.splat_backend not in ('auto', 'torch', 'cupy'):
            raise ValueError('splat_backend must be auto/torch/cupy')

