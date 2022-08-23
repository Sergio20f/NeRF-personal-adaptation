import yaml


class Config:

    def __init__(self, filepath):
        self.filepath = filepath
        with open(filepath, "r") as file_descriptor:
            data = yaml.load(file_descriptor, Loader=yaml.Loader)

        self.data = data

        # Paths attributes
        dataset = data["paths"]["dataset"]
        self.DATASET = dataset

        train_json = data["paths"]["train_json"]
        self.TRAIN_JSON = train_json

        val_json = data["paths"]["val_json"]
        self.VAL_JSON = val_json

        test_json = data["paths"]["test_json"]
        self.TEST_JSON = test_json

        img_path = data["paths"]["img_path"]
        self.IMG_PATH = img_path

        coarse_path = data["paths"]["coarse_path"]
        self.COARSE_PATH = coarse_path

        fine_path = data["paths"]["fine_path"]
        self.FINE_PATH = fine_path

        video_path = data["paths"]["video_path"]
        self.VIDEO_PATH = video_path

        # Image features
        img_width = data["img_features"]["img_width"]
        self.IMG_WIDTH = img_width

        img_height = data["img_features"]["img_height"]
        self.IMG_HEIGHT = img_height

        near = data["img_features"]["near"]
        self.NEAR = near

        far = data["img_features"]["far"]
        self.FAR = far

        # Train
        batch_size = data["train"]["batch_size"]
        self.BATCH_SIZE = batch_size

        l_coor = data["train"]["l_coor"]
        self.L_COOR = l_coor

        l_dir = data["train"]["l_dir"]
        self.L_DIR = l_dir

        dense_units = data["train"]["dense_units"]
        self.DENSE_UNITS = dense_units

        skip_layer = data["train"]["skip_layer"]
        self.SKIP_LAYER = skip_layer

        n_c = data["train"]["n_c"]
        self.N_C = n_c

        n_f = data["train"]["n_f"]
        self.N_F = n_f

        steps_per_epoch = data["train"]["steps_per_epoch"]
        self.STEPS_PER_EPOCH = steps_per_epoch

        validation_steps = data["train"]["validation_steps"]
        self.VALIDATION_STEPS = validation_steps

        epochs = data["train"]["epochs"]
        self.EPOCHS = epochs

        # Inference
        sample_theta_points = data["inference"]["sample_theta_points"]
        self.SAMPLE_THETA_POINTS = sample_theta_points

        # Video
        fps = data["video"]["fps"]
        self.FPS = fps

        quality = data["video"]["quality"]
        self.QUALITY = quality

        macro_block_size = data["video"]["macro_block_size"]
        self.MACRO_BLOCK_SIZE = macro_block_size

        pass
    # Data input pipeline


config = Config("config.yaml")