import AIAG_loss
import AADB_AIAG_loss
def load(args):
    if args.id in ['AIAG']:
        criterion = AIAG_loss.AIAG_Loss_1(args)
    elif args.id in ['AADB_AIAG', 'AADB_AIAG_RGB']:
        criterion = AADB_AIAG_loss.AIAG_Loss_1(args)
    else:
        print('Wrong ID specified in Loss: %s'%(args.id))
        exit()
    return criterion
    


            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            