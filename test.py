from utils import img_c2w, read_json
from config_loader import config


json_train = read_json(config.TRAIN_JSON)
print(img_c2w(json_train, config.DATASET_TRAIN)[0])
