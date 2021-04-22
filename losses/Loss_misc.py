import AIAG_loss

def load(args):
    if args.id in ['AIAG']:
        criterion = AIAG_loss.AIAG_Loss_1(args)
    else:
        print('Wrong ID specified in Loss: %s'%(args.id))
        exit()
    return criterion
    


            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            