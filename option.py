import argparse

def get_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str,
                        default='/media/s1/cjg/dataset/GRP/TRAIN',
                        help='root directory of the dataset')
    parser.add_argument('--positives_per_query', type=int, default=8,
                        help='Number of potential positives in each training tuple')
    parser.add_argument('--negatives_per_query', type=int, default=8,
                        help='Number of definite negatives in each training tuple')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='Number of each point cloud')
    parser.add_argument('--output_dim', type=int, default=256,
                        help='Dim of desc output')
    parser.add_argument('--emb_dims', type=int, default=1024,
                        help='Dim of features')
    parser.add_argument('--margin_1', type=float, default=0.5,
                        help='Margin for hinge loss')
    parser.add_argument('--margin_2', type=float, default=0.2,
                        help='Margin for hinge loss')
    parser.add_argument('--loss_lazy', action='store_false',default=True,
                        help='If present, do not use lazy variant of loss')
    parser.add_argument('--triplet_use_best_positives', action='store_false',default=True,
                        help='If present, use best positives, otherwise use hardest positives')
    parser.add_argument('--loss_ignore_zero_batch', action='store_true',default=False,
                        help='If present, mean only batches with loss > 0.0')
    parser.add_argument('--voxel_resolution', type=float, default=0.1,
                        help='The voxel size of downsample')
    parser.add_argument('--filter_ground', action='store_true',default=False,
                        help='If present, mean filter ground pointcloud')

    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--num_epochs', type=int, default=400,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')

    parser.add_argument('--ckpt_path', type=str, default='',
                        help='pretrained checkpoint path to load')
    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss'],
                        help='the prefixes to ignore in the checkpoint state dict')

    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer type',
                        choices=['sgd', 'adam', 'radam', 'ranger'])
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='steplr',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')
    ###########################
    #### params for steplr ####
    parser.add_argument('--decay_step', nargs='+', type=int, default=[20],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.1,
                        help='learning rate decay amount')
    ###########################
    #### params for poly ####
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='exponent for polynomial learning rate decay')
    ###########################

    parser.add_argument('--use_amp', default=False, action="store_true",
                        help='use mixed precision training (NOT SUPPORTED!)')

    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')

    return parser.parse_args()
