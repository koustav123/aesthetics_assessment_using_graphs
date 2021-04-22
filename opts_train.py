import argparse

def parse_opts():
    parser = argparse.ArgumentParser(description='Aesthetics MTL')

    #Input files                        
    parser.add_argument('--db', type=str, default='',
                        help='location of the pickle/json with imid and labels')
    parser.add_argument('--datapath', type=str, default='',
                        help='location of the data corpus')
    parser.add_argument('--feature_path', type=str, default='',
                        help='location of the features')

    
    #Standard  Hyperparameters during training       
    parser.add_argument('--aug', type=str, default = 'MLSP_8_NUMPY',
                    help='data augmentation strategy')
    parser.add_argument('--aug_target', type=str, default = 'NO_TARGET_TRANSFORM',
                    help='target augmentation strategy')
    parser.add_argument('--num_epochs', type=int, default=50,
                    help='upper epoch limit')
    # parser.add_argument('--lr_decay_after', type=int, default=1,
    #                 help='start reducing learning rate after this number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001,
                    help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default = 32, metavar='N',
                    help='batch size')
    parser.add_argument('--batch_size_test', type=int, default=128, metavar='N',
                        help='batch size_test')
    parser.add_argument('--optimizer', type=str, default = 'ADAM',
                    help='ADAM or SGD')
    parser.add_argument('--pilot', type=int, default=-1, metavar='N',
                        help='pilot dataset size')
    parser.add_argument('--n_workers', type=int, default = 0, metavar='N',
                    help='workers for multiprocessing')
    parser.add_argument('--w_emd', type=float, default=0,
                    help='emd loss weight for A2')
    parser.add_argument('--w_mse', type=float, default=0,
                        help='mse loss weight for A2')



    #Architecture specifications
    parser.add_argument('--A2_D', type=int, default=10,
                        help='A2 o/p 10 or 1')
    parser.add_argument('--start_from', type=str, default=None,
                        help='location of the pre-trained_model')
    parser.add_argument('--base_model', type=str, default="-1",
                        help='basemodel')
    parser.add_argument('--data_precision', type=int, default=-1,
                        help='Floating point precision to use')
    parser.add_argument('--model_name', type=str, default="Avg_Pool_FC",
                        help='name of the pre-trained_model')


    #Checkpointing and visualization
    parser.add_argument('--save', type=str,  default='',
                    help='path to save model')
    parser.add_argument('--save_visuals', type=str,  default='runs/',
                    help='path to save visuals')
    parser.add_argument('--id', type=str, default='',
                        help='id to use')
    parser.add_argument('--exp_id', type=str, default='',
                        help='experiment id to use')
    parser.add_argument('--val_after_every', type=int, default = 1, help='validate after every N epochs' )

    args = parser.parse_args()

    return args
