from commons import package_utils

tf_version = package_utils.get_tf_major_version()
if tf_version == 1:
    from keras.models import Sequential
else:
    from tensorflow.keras.models import Sequential

def load_model_weights(model: Sequential, weight_file: str) -> Sequential:
    """
    Load pre-trained weights for a given model
    Args:
        model (keras.models.Sequential): pre-built model
        weight_file (str): exact path of pre-trained weights
    Returns:
        model (keras.models.Sequential): pre-built model with
            updated weights
    """
    try:
        model.load_weights(weight_file)
    except Exception as err:
        raise ValueError(
            f"An exception occurred while loading the pre-trained weights from {weight_file}."
            "This might have happened due to an interruption during the download."
            "You may want to delete it and allow DeepFace to download it again during the next run."
            "If the issue persists, consider downloading the file directly from the source "
            "and copying it to the target folder."
        ) from err
    return model