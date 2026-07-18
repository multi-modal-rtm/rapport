NUM_FRAMES = 8
FRAME_SIZE = 224
SAMPLE_RATE = 16000

# Kinetics-400 normalization stats used by torchvision's MViTv2 weights.
FRAME_MEAN = (0.45, 0.45, 0.45)
FRAME_STD = (0.225, 0.225, 0.225)

EMOTION_LABELS = ["neutral", "joy", "sadness", "anger", "surprise", "fear", "disgust"]
EMOTION2ID = {label: i for i, label in enumerate(EMOTION_LABELS)}

# IEMOCAP, 6-class protocol (docs/RECIPE.md's "IEMOCAP -- label protocol
# decision" section, Phase N5-B Step B2): {angry, happy, excited, sad,
# neutral, frustrated}. Order fixed here once and never reordered --
# downstream label-id-dependent code (confusion matrices, per-class F1
# dicts) assumes this exact ordering.
IEMOCAP_EMOTION_LABELS = ["angry", "happy", "excited", "sad", "neutral", "frustrated"]
IEMOCAP_EMOTION2ID = {label: i for i, label in enumerate(IEMOCAP_EMOTION_LABELS)}
# Raw EmoEvaluation 3-letter consensus codes -> canonical label name.
IEMOCAP_CODE2LABEL = {"ang": "angry", "hap": "happy", "exc": "excited", "sad": "sad", "neu": "neutral", "fru": "frustrated"}
