import argparse
from models.config import ModelConfig


def add_model_arguments(parser, suppress_defaults=False):
    cfg = ModelConfig()
    default = lambda key: argparse.SUPPRESS if suppress_defaults else getattr(cfg, key)
    parser.add_argument('--voxel_bins', type=int, default=default('voxel_bins'))
    parser.add_argument('--nb_of_flow', type=int, default=default('nb_of_flow'))
    parser.add_argument('--refine', choices=['resblock','transformer'], default=default('refine'))
    parser.add_argument('--unit_dim', type=int, choices=[32,44], default=default('unit_dim'))
    for key in ('legacy_query','legacy_warp','mask_in_refine','basr','flow_corr','splat_metric',
                'checkpoint_bifeb','checkpoint_refine'):
        parser.add_argument('--'+key, action=argparse.BooleanOptionalAction, default=default(key))
    parser.add_argument('--basr_heads', type=int, nargs=2, default=default('basr_heads'))
    parser.add_argument('--basr_chunk_size', type=int, default=default('basr_chunk_size'))
    parser.add_argument('--basr_support_ratio', type=float, default=default('basr_support_ratio'))
    parser.add_argument('--basr_keys', choices=['warped'], default=default('basr_keys'))
    parser.add_argument('--correlation_radius', type=int, default=default('correlation_radius'))
    parser.add_argument('--splat_backend', choices=['auto','torch','cupy'], default=default('splat_backend'))
    parser.add_argument('--w_flow', type=float, default=default('w_flow'))
    parser.add_argument('--w_ps', type=float, default=default('w_ps'))


def add_data_arguments(parser, evaluation=False):
    parser.add_argument('--dataset', choices=['gopro','bsergb'], default='bsergb' if evaluation else 'gopro')
    parser.add_argument('--data_root','--test_root', dest='data_root')
    parser.add_argument('--skips', type=int, nargs='+', default=None)
    parser.add_argument('--legacy_lpips', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--num_targets', type=int, default=None,
                        help='optional training target cap; omitted means all intermediate frames, as in OF-AF')
    parser.add_argument('--crop_size', type=int, default=256)
