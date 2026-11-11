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
    parser.add_argument('--frame_timestamps', default=None, help='per-scene frame timestamp file; autodetected if omitted')
    parser.add_argument('--image_fps', type=float, default=None, help='known RGB FPS, default 240 only for GoPro')
    parser.add_argument('--time_offset', type=float, default=0., help='seconds, used with FPS-generated frame times')
    parser.add_argument('--frame_time_scale', type=float, default=1., help='convert frame-file timestamps to seconds')
    parser.add_argument('--event_time_scale', type=float, default=1., help='convert raw event and event-file timestamps to seconds')
    parser.add_argument('--event_coords', choices=['ofaf','pixels'], default='ofaf')
    parser.add_argument('--polarity', choices=['auto','01','pm1'], default='auto')
    parser.add_argument('--legacy_event_time', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--legacy_lpips', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--scene_file', default=None)
    parser.add_argument('--num_targets', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--eval_stride', type=int, default=None)
    parser.add_argument('--gopro_odd_targets', action=argparse.BooleanOptionalAction, default=False,
                        help='evaluate zero-based odd target indices as upstream; pair with original stride/protocol')
    parser.add_argument('--include_bad_frames', action='store_true')

