import torch
import Inception_ResNet_V2_Base, Graph_Models
from termcolor import colored

def load(args, dset_classes):
    if args.id in ['AIAG', 'AIAG_Extraction']:
        if args.base_model == 'inceptionresnetv2':
            if args.id in ['AIAG_Extraction']:
                model = Inception_ResNet_V2_Base.inceptionresnetv2()
            else:
                # Specify the particular baseline model
                model = getattr(Graph_Models, args.model_name)(args)
        else:
            print('Wrong model specified: %s' % (args.basemodel))
            exit(1)
    else:
        print('Wrong id specified: %s' % (args.id))
        exit(1)

    if args.start_from != None:
        model.load_state_dict(torch.load(args.start_from)['model'])
        print(colored('PTM Loaded from: %s' % args.start_from, 'white'))
    if args.data_precision == 16:
        return model.half().cuda()
    elif args.data_precision == 32:
        return model.float().cuda()
    else:
        print('Wrong precision specified: %s' % (args.data_precision))
        exit(1)
