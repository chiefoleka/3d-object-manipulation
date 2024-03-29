from consts.model_consts import ModelType
from models.basic import BasicModel, BasicModelLarge, BasicModelExtraLarge
from models.basic_norm import BasicModelNorm, BasicModelNormLarge, BasicModelNormExtraLarge


class ModelFactory:
    # defaults to BasicModel
    @staticmethod
    def getModel(className, num_node_features = 3, num_classes = 1, device = None):
        if className == ModelType.BASIC_MODEL_LARGE:
            return BasicModelLarge(num_node_features, num_classes).to(device)
        elif className == ModelType.BASIC_MODEL_NORM:
            return BasicModelNorm(num_node_features, num_classes).to(device)
        elif className == ModelType.BASIC_MODEL_NORM_LARGE:
            return BasicModelNormLarge(num_node_features, num_classes).to(device)
        elif className == ModelType.BASIC_MODEL_EXTRA_LARGE:
            return BasicModelExtraLarge(num_node_features, num_classes).to(device)
        elif className == ModelType.BASIC_MODEL_NORM_EXTRA_LARGE:
            return BasicModelNormExtraLarge(num_node_features, num_classes).to(device)
        else:
            return BasicModel(num_node_features, num_classes).to(device)