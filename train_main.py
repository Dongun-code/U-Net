from model.model_factory import model_factory
from config import Config as cfg
from model.model_factory import model_factory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def main():
    model = model_factory(cfg.Model.model, cfg.Model.input_shape, cfg.Model.batch_size).get_model()
    model.fit()


if __name__ == "__main__":
    main()