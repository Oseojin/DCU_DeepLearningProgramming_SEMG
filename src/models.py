"""CWT 분류기 후보. 모든 모델은 무작위 초기화에서 시작한다(논문과 동일 조건)."""

from torch import nn
from torchvision import models
from torchvision.models.densenet import DenseNet

NUM_CLASSES = 5

MODEL_NAMES = (
    "densenet_s_cwt", "densenet_xs_cwt", "densenet121_cwt", "densenet121",
    "densenet161", "densenet161_cwt", "resnet18", "resnet18_cwt",
    "efficientnet_b0", "mobilenet_v3_small",
)

# (3, 32, 300) 입력에서 스케일축(32)을 stem에서 4배로 줄이지 않는 설정.
# 기본 stem: 7x7 stride 2 + 3x3 maxpool stride 2 -> 32가 곧바로 8이 된다.
CWT_STEM = {"kernel_size": (3, 7), "stride": (1, 2), "padding": (1, 3), "bias": False}
CWT_POOL = {"kernel_size": (2, 3), "stride": (2, 2), "padding": (0, 1)}

# name -> (growth_rate, block_config, num_init_features)
COMPACT = {
    "densenet_s_cwt": (24, (6, 12, 18, 12), 48),
    "densenet_xs_cwt": (16, (4, 8, 12, 8), 32),
}


def _apply_cwt_stem(model, out_channels):
    model.features.conv0 = nn.Conv2d(3, out_channels, **CWT_STEM)
    nn.init.kaiming_normal_(model.features.conv0.weight, mode="fan_out", nonlinearity="relu")
    model.features.pool0 = nn.MaxPool2d(**CWT_POOL)
    return model


def make_model(name="densenet_s_cwt", dropout=0.2, drop_rate=0.0):
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model: {name}. Choose from {MODEL_NAMES}")
    if not 0 <= dropout < 1 or not 0 <= drop_rate < 1:
        raise ValueError("dropout and drop_rate must be in [0, 1)")

    if name in COMPACT:
        growth, blocks, init_features = COMPACT[name]
        model = DenseNet(growth_rate=growth, block_config=blocks,
                         num_init_features=init_features, num_classes=NUM_CLASSES,
                         drop_rate=drop_rate)
        _apply_cwt_stem(model, init_features)
    elif name.startswith("densenet"):
        builder = models.densenet161 if name.startswith("densenet161") else models.densenet121
        model = builder(weights=None, num_classes=NUM_CLASSES, drop_rate=drop_rate)
        if name.endswith("_cwt"):
            _apply_cwt_stem(model, 96 if name.startswith("densenet161") else 64)
    elif name.startswith("resnet18"):
        model = models.resnet18(weights=None, num_classes=NUM_CLASSES)
        if name.endswith("_cwt"):
            model.conv1 = nn.Conv2d(3, 64, **CWT_STEM)
            nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
            model.maxpool = nn.MaxPool2d(**CWT_POOL)
        model.fc = nn.Sequential(nn.Dropout(dropout), model.fc)
        return model
    else:
        return getattr(models, name)(weights=None, num_classes=NUM_CLASSES, dropout=dropout)

    model.classifier = nn.Sequential(nn.Dropout(dropout), model.classifier)
    return model


def decay_groups(model, weight_decay):
    """bias와 정규화 계층의 scale/shift에는 weight decay를 적용하지 않는다."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = no_decay if parameter.ndim <= 1 or name.endswith(".bias") else decay
            group.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
