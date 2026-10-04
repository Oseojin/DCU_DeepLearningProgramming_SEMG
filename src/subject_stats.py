"""사람(클래스)별 원신호 통계. 혼동행렬 분석에서 '왜 이 쌍이 헷갈리는가'를 볼 때 쓴다.

  python subject_stats.py                     # ../data 의 CSV 250개를 읽는다

APB(채널 1)와 ADM(채널 2)의 RMS, 그리고 시행별 APB/ADM RMS 비의 평균을 사람별로 낸다.
배포 CSV는 이미 필터링되어 있으므로 추가 필터링 없이 그대로 쓴다. numpy만 필요하다.
"""

import argparse
from pathlib import Path

import numpy as np

SUBJECTS = "ABCDE"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--output", type=Path, help="결과를 markdown 표로도 저장")
    args = parser.parse_args()

    lines = ["| 사람 | 시행 수 | APB RMS | ADM RMS | 두 채널 RMS 평균 | APB/ADM 비 (시행별 평균) |",
             "|---|---:|---:|---:|---:|---:|"]
    for subject in SUBJECTS:
        files = sorted((args.data_dir / "data" / subject).glob("*.csv"))
        rms = np.array([np.sqrt((np.loadtxt(f, delimiter=",", skiprows=1) ** 2).mean(axis=0))
                        for f in files])
        lines.append(f"| {subject} | {len(files)} | {rms[:, 0].mean():.4f} | {rms[:, 1].mean():.4f} | "
                     f"{rms.mean():.4f} | {(rms[:, 0] / rms[:, 1]).mean():.2f} |")
    text = "\n".join(lines)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
