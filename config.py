"""Configuration of hyper-parameters.

See https://www.kaggle.com/dqhdqmcttdqx/carvana-challenge-unet#Config for details.
"""

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 2e-5
DEVICE = 'cpu'
BATCH_SIZE = 32
TEST_BATCH_SIZE = 128
NUM_EPOCHS = 50
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False