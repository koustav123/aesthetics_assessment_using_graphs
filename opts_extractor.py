import argparse

def parse_opts():
    parser = argparse.ArgumentParser(description='Aesthetics')

    #Input files                        
    parser.add_argument('--db', type=str, default='',
                        help='location of the pickle/json with imid and labels')
    parser.add_argument('--datapath', type=str, default='',
                        help='location of the data corpus')
    parser.add_argument('--feature_path', type=str, default=None,
                        help='location of the features')
    parser.add_argument('--feature_file_name', type=str, default='',
                        help='output_file_name')
    parser.add_argument('--save_feat', type=str, default='',
                        help='path to save features extracted')

    #Standard  Hyperparameters during training
    parser.add_argument('--aug', type=str, default = 'MLSP_AUG_FULL_IMAGE',
                    help='data augmentation strategy')
    parser.add_argument('--aug_target', type=str, default = 'NO_TARGET_TRANSFORM',
                    help='target augmentation strategy')
    parser.add_argument('--num_epochs', type=int, default=1,
                    help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default = 1, metavar='N',
                    help='batch size')
    parser.add_argument('--batch_size_test', type=int, default=1, metavar='N',
                        help='batch size_test')
    parser.add_argument('--pilot', type=int, default=1000, metavar='N',
                        help='pilot dataset size')
    parser.add_argument('--n_workers', type=int, default = 0, metavar='N',
                    help='workers for multiprocessing')


    #Architecture specifications
    parser.add_argument('--pretrained', action='store_true',
                    help='use pretrained model')
    parser.add_argument('--A2_D', type=int, default=10,
                        help='A2 o/p 10 or 1')

    parser.add_argument('--start_from', type=str, default=None,
                        help='location of the pre-trained_model')
    parser.add_argument('--base_model', type=str, default="-1",
                        help='basemodel')
    parser.add_argument('--data_precision', type=int, default=-1,
                        help='Floating point precision to use')


    #Checkpointing and visualization
    parser.add_argument('--id', type=str, default='AIAG',
                        help='id to use')
    parser.add_argument('--exp_id', type=str, default='',
                        help='experiment id to use')

    args = parser.parse_args()

    return args
