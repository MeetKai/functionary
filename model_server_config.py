
import modal

MODEL = "meetkai/functionary-small-v2.4"
#MODEL = "meetkai/functionary-medium-v2.4"
MAX_MODEL_LENGTH = 8196
LOADIN8BIT = False
GPU_CONFIG = modal.gpu.L4(count=1)
#GPU_CONFIG = modal.gpu.A100(memory=80, count=2)
MODEL_DIR = "/model2"
GRAMMAR_SAMPLING = False