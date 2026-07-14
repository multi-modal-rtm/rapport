NUM_FRAMES = 8
FRAME_SIZE = 224
SAMPLE_RATE = 16000

# Kinetics-400 normalization stats used by torchvision's MViTv2 weights.
FRAME_MEAN = (0.45, 0.45, 0.45)
FRAME_STD = (0.225, 0.225, 0.225)

EMOTION_LABELS = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]
EMOTION2ID = {label: i for i, label in enumerate(EMOTION_LABELS)}
