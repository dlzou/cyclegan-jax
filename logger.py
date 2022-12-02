import logging
import sys

logging.basicConfig(filename='traingging.txt',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

# This writes to the output in addition to writing to a file 
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

logging.info("Running CycleGAN logger")

logger = logging.getLogger('cycleGAN')

