# Build model from scratch and train it
# MODEL_TYPE="simple_cnn"
# TRAIN_FOLDER="datasets/train/input"
# TRAIN_GROUND_TRUTH_FOLDER="datasets/train/GT"
# MODEL_SAVED_FOLDER=saved_models/$MODEL_TYPE
# IMAGE_WIDTH=640
# IMAGE_HEIGHT=640
# NUM_CLASSES=6
# BATCH_SIZE=8
# EPOCHS=100
# LR=0.0001

# python train.py \
#     --model_type $MODEL_TYPE \
#     --train_folder $TRAIN_FOLDER \
#     --train_ground_truth_folder $TRAIN_GROUND_TRUTH_FOLDER \
#     --model_saved_folder $MODEL_SAVED_FOLDER \
#     --image_width $IMAGE_WIDTH \
#     --image_height $IMAGE_HEIGHT \
#     --num_classes $NUM_CLASSES \
#     --batch_size $BATCH_SIZE \
#     --epochs $EPOCHS \
#     --lr $LR

# Load pretrained model and train it
# MODEL_TYPE="deeplabv3_resnet50_pretrained"
# TRAIN_FOLDER="datasets/train/input"
# TRAIN_GROUND_TRUTH_FOLDER="datasets/train/GT"
# MODEL_SAVED_FOLDER=saved_models/$MODEL_TYPE
# IMAGE_WIDTH=640
# IMAGE_HEIGHT=640
# NUM_CLASSES=6
# BATCH_SIZE=8
# EPOCHS=50
# LR=0.0001

# python train.py \
#     --model_type $MODEL_TYPE \
#     --train_folder $TRAIN_FOLDER \
#     --train_ground_truth_folder $TRAIN_GROUND_TRUTH_FOLDER \
#     --model_saved_folder $MODEL_SAVED_FOLDER \
#     --image_width $IMAGE_WIDTH \
#     --image_height $IMAGE_HEIGHT \
#     --num_classes $NUM_CLASSES \
#     --batch_size $BATCH_SIZE \
#     --epochs $EPOCHS \
#     --lr $LR

# Use pretrained model architecture and train it from scratch
MODEL_TYPE="deeplabv3_resnet50_scratch"
TRAIN_FOLDER="datasets/train/input"
TRAIN_GROUND_TRUTH_FOLDER="datasets/train/GT"
MODEL_SAVED_FOLDER=saved_models/$MODEL_TYPE
IMAGE_WIDTH=640
IMAGE_HEIGHT=640
NUM_CLASSES=6
BATCH_SIZE=8
EPOCHS=50
LR=0.0001

python train.py \
    --model_type $MODEL_TYPE \
    --train_folder $TRAIN_FOLDER \
    --train_ground_truth_folder $TRAIN_GROUND_TRUTH_FOLDER \
    --model_saved_folder $MODEL_SAVED_FOLDER \
    --image_width $IMAGE_WIDTH \
    --image_height $IMAGE_HEIGHT \
    --num_classes $NUM_CLASSES \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lr $LR