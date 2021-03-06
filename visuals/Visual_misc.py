import datetime
import  AIAG_Visualizer
import  AADB_AIAG_Visualizer

def load(args, dset_loaders = None):

    if args.id in ['AIAG']:
        visualizer = AIAG_Visualizer.A2_Visualizer(comment = args.save_visuals + '/Exp: ' + args.exp_id + \
            ' Tasks: ' + args.id + ' -- ' +datetime.datetime.now().strftime("%I:%M%p %B %d, %Y"), args = args)
    elif args.id in ['AADB_AIAG', 'AADB_AIAG_RGB']:
        visualizer = AADB_AIAG_Visualizer.A2_Visualizer(comment = args.save_visuals + '/Exp: ' + args.exp_id + \
            ' Tasks: ' + args.id + ' -- ' +datetime.datetime.now().strftime("%I:%M%p %B %d, %Y"), args = args)
    else:
        print('Wrong TaskID passed to Visualizer: %s'%(args.id))
        exit()
    return visualizer
    
    


