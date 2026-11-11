import sys
from bsergb_eval import main

if __name__=='__main__':
    main(['--dataset','gopro']+sys.argv[1:])

