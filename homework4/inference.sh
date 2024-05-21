MODEL_TYPE="deeplabv3_resnet50_pretrained"
# Available options: deeplabv3_resnet50_pretrained, deeplabv3_resnet50_scratch, simple_cnn
IMAGE_WIDTH=640
IMAGE_HEIGHT=640
NUM_CLASSES=6
WEIGHT_PATH=saved_models/deeplabv3_resnet50_pretrained/49.pth
IMAGE_FOLDER="datasets/test/input"
GROUND_TRUTH_FOLDER="datasets/test/GT"
PREDICTION_FOLDER=datasets/test/$MODEL_TYPE"_49"

python inference.py \
    --image_folder $IMAGE_FOLDER \
    --ground_truth_folder $GROUND_TRUTH_FOLDER \
    --prediction_folder $PREDICTION_FOLDER \
    --image_width $IMAGE_WIDTH \
    --image_height $IMAGE_HEIGHT \
    --num_classes $NUM_CLASSES \
    --weight_path $WEIGHT_PATH \
    --model_type $MODEL_TYPE