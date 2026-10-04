"""CWT 스펙트로그램 전용 증강. 이미지 증강(회전/좌우반전/색상)은 의도적으로 제외한다.

시간축(300)과 스케일축(32)은 의미가 다르므로 축마다 다른 연산을 적용한다.
모든 함수는 (B, 3, 32, 300) 텐서를 받고 같은 모양을 돌려준다.
"""

import numpy as np
import torch


def random_time_roll(x, max_shift):
    """윈도우 경계 위치에 대한 불변성을 학습시킨다. 배치 내 샘플마다 다른 이동량."""
    if max_shift <= 0:
        return x
    shifts = torch.randint(-max_shift, max_shift + 1, (x.shape[0],), device=x.device)
    # roll은 샘플별 이동량을 지원하지 않으므로 gather로 처리한다.
    index = torch.arange(x.shape[3], device=x.device).unsqueeze(0) - shifts.unsqueeze(1)
    index = index % x.shape[3]
    index = index.view(x.shape[0], 1, 1, x.shape[3]).expand_as(x)
    return x.gather(3, index)


def spec_augment(x, time_masks=2, time_width=30, scale_masks=2, scale_width=4, p=0.5):
    """SpecAugment. 시간축은 넓게, 스케일축은 좁게 가린다(스케일 해상도가 32뿐이므로)."""
    if p <= 0:
        return x
    batch, _, scales, steps = x.shape
    x = x.clone()
    apply = torch.rand(batch, device=x.device) < p
    rows = torch.arange(scales, device=x.device).view(1, scales, 1)
    cols = torch.arange(steps, device=x.device).view(1, 1, steps)
    mask = torch.zeros(batch, scales, steps, dtype=torch.bool, device=x.device)
    for width, count, axis in ((time_width, time_masks, "t"), (scale_width, scale_masks, "s")):
        if width <= 0 or count <= 0:
            continue
        limit = steps if axis == "t" else scales
        for _ in range(count):
            span = torch.randint(0, width + 1, (batch, 1, 1), device=x.device)
            start = (torch.rand(batch, 1, 1, device=x.device) * (limit - span).clamp(min=1)).long()
            grid = cols if axis == "t" else rows
            mask |= (grid >= start) & (grid < start + span)
    mask &= apply.view(batch, 1, 1)
    return x.masked_fill(mask.unsqueeze(1), 0.0)


def channel_gain(x, low=0.9, high=1.1):
    """전극 임피던스 차이를 모사한다. 3번째 채널은 앞 두 채널의 평균이므로 재계산한다."""
    if low >= high:
        return x
    gain = torch.empty(x.shape[0], 2, 1, 1, device=x.device).uniform_(low, high)
    first = x[:, :2] * gain
    return torch.cat([first, first.mean(dim=1, keepdim=True)], dim=1)


def gaussian_noise(x, sigma):
    return x if sigma <= 0 else x + torch.randn_like(x) * sigma


def apply_augmentations(x, config):
    """학습 배치에만 적용한다. 순서: 게인 -> 시간이동 -> 잡음 -> 마스킹."""
    x = channel_gain(x, config["gain_low"], config["gain_high"])
    x = random_time_roll(x, config["time_roll"])
    x = gaussian_noise(x, config["noise_sigma"])
    x = spec_augment(x, config["time_masks"], config["time_width"],
                     config["scale_masks"], config["scale_width"], config["spec_p"])
    return x


def mixup(x, y, alpha, generator=None):
    """반환: 섞인 입력, 원본 라벨, 섞인 라벨, 혼합 비율. alpha<=0이면 통과."""
    if alpha <= 0:
        return x, y, y, 1.0
    lam = float(np.random.default_rng(
        None if generator is None else int(torch.randint(0, 2**31 - 1, (1,), generator=generator))
    ).beta(alpha, alpha))
    lam = max(lam, 1.0 - lam)  # 원본 라벨이 항상 주 라벨이 되도록 한다.
    order = torch.randperm(x.shape[0], device=x.device)
    return lam * x + (1.0 - lam) * x[order], y, y[order], lam


def mixup_loss(criterion, logits, y_a, y_b, lam):
    if lam >= 1.0:
        return criterion(logits, y_a)
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


DEFAULT_AUGMENT = {
    "gain_low": 1.0, "gain_high": 1.0, "time_roll": 0, "noise_sigma": 0.0,
    "time_masks": 0, "time_width": 0, "scale_masks": 0, "scale_width": 0, "spec_p": 0.0,
}

STRONG_AUGMENT = {
    "gain_low": 0.9, "gain_high": 1.1, "time_roll": 30, "noise_sigma": 0.02,
    "time_masks": 2, "time_width": 30, "scale_masks": 2, "scale_width": 4, "spec_p": 0.5,
}
