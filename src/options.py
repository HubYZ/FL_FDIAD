import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--frac', type=float, default=0.75)
    parser.add_argument('--local_ep', type=int, default=5)
    parser.add_argument('--local_bs', type=int, default=48)
    parser.add_argument('--data_dir',type=str, default=r'/scratch/er95/xy5751/1-HV-mixed--0-no_sw_subGrid_aligned')

    parser.add_argument('--model_name', type=str, default='FL_1-HV-mixed--0-no_sw_subGrid')
    parser.add_argument('--JOBID', type=str, default='000000.gadi-pbs')
    parser.add_argument('--model_path', type=str, default='.')

    # local clients
    parser.add_argument('--col_bus', type=int, default=3)
    parser.add_argument('--col_line', type=int, default=3)
    parser.add_argument('--in_channel', type=int, default=1)
    parser.add_argument('--channel_axis', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--gpu_num', type=int, default=4)
    parser.add_argument('--seq_size', type=int, default=96)
    parser.add_argument('--report_freq', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--grad_clip', type=float, default=5.0)
    parser.add_argument('--learning_rate', type=float, default=0.001)

    args = parser.parse_args()
    return args
