# Loading model weights

# In this file, we define download_model()
#   which will be called in the Dockerfile

# Alternatively,
#   can fetch the model weights in the Dockerfile (S3, git-lfs, etc.)
#   and save them into cache directories (e.g. ~/.cache/torch/transformers/)


# WARNING ⚠️ Ultralytics settings reset to defaults.
# This is normal and may be due to a recent ultralytics package update, but may have overwritten previous settings.
# You may view and update settings directly in '/Users/drslimm/Library/Application Support/Ultralytics/settings.yaml'

# https://huggingface.co/keremberke/yolov8m-table-extraction


from ultralyticsplus import YOLO


def download_model():
    print("Downloading weights...\n")
    model = YOLO('keremberke/yolov8m-table-extraction')
    print("Weights downloaded.\n")


if __name__ == "__main__":
    download_model()
