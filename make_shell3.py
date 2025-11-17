#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
把 AISHELL-3 预处理成一个 CSV:
audio, transcription, speaker_id, gender, accent, utt_id

- 使用的文件：
  - /data/Teammember/jinyang.he/workspace/datasets/train/content.txt
  - /data/Teammember/jinyang.he/workspace/spk-info.txt
  - /data/Teammember/jinyang.he/workspace/datasets/train/wav/**.wav
"""

from pathlib import Path
import csv

# ================== 按你的真实路径设置 ==================
# AISHELL-3 train 目录（里面有 content.txt 和 wav/）
DATASET_ROOT = Path("/data/Teammember/jinyang.he/workspace/datasets/train")

CONTENT_PATH = DATASET_ROOT / "content.txt"
WAV_ROOT = DATASET_ROOT / "wav"

# spk-info.txt 在 workspace 目录下
SPK_INFO_PATH = DATASET_ROOT.parent / "spk-info.txt"

OUT_CSV = Path("aishell3_for_parler.csv")
# =======================================================


def is_pinyin_token(tok: str) -> bool:
    """
    判断一个 token 是否是拼音（带声调数字）。
    规则：只包含字母和数字（纯 ASCII），视为拼音。
    例：guang3, nv3, wanr2, hair2 -> True
        广, 州, 。 -> False
    """
    if not tok:
        return False
    # 只要有非 ASCII 字符，就不是拼音
    if not tok.isascii():
        return False
    # ASCII 里如果只含字母 / 数字，则当作拼音
    return all(ch.isalpha() or ch.isdigit() for ch in tok)


def load_utt2text_from_content(content_path: Path):
    """
    解析 content.txt：
    每行形如：
    SSB00050001.wav<TAB>广 guang3 州 zhou1 女 nv3 ...
    -> 返回 { "SSB00050001.wav": "广州女大学生登山失联四天警方找到疑似女尸", ... }
    """
    utt2text = {}
    if not content_path.exists():
        raise FileNotFoundError(f"content.txt not found at: {content_path}")

    with open(content_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # 优先按 TAB 切
            if "\t" in line:
                fname, rest = line.split("\t", 1)
            else:
                # 兼容：文件名 + 空格 + 文本的情况
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    print("跳过无法解析的行:", line)
                    continue
                fname, rest = parts

            tokens = rest.strip().split()
            hanzi_tokens = [t for t in tokens if not is_pinyin_token(t)]
            text = "".join(hanzi_tokens)

            utt2text[fname] = text

    print(f"[INFO] 从 content.txt 解析到 {len(utt2text)} 条文本")
    return utt2text


def load_speaker_meta(spk_info_path: Path):
    """
    解析 spk_info.txt：
    SSB1837  B  female  north
    -> 返回 { "SSB1837": {"gender": "female", "accent": "north"}, ... }
    """
    spk_meta = {}
    if not spk_info_path.exists():
        print(f"[WARN] spk-info.txt 未找到：{spk_info_path}，gender/accent 将留空")
        return spk_meta

    with open(spk_info_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            spk_id, age_group, gender, accent = parts[:4]
            spk_meta[spk_id] = {
                "gender": gender,
                "accent": accent,
                "age_group": age_group,
            }

    print(f"[INFO] 从 spk-info.txt 解析到 {len(spk_meta)} 个说话人")
    return spk_meta


def main():
    if not WAV_ROOT.exists():
        raise FileNotFoundError(f"wav 目录不存在: {WAV_ROOT}")

    utt2text = load_utt2text_from_content(CONTENT_PATH)
    spk_meta = load_speaker_meta(SPK_INFO_PATH)

    rows = []
    missing_text = 0
    missing_spk_meta = 0

    # 遍历所有 wav 文件
    for wav_path in WAV_ROOT.rglob("*.wav"):
        fname = wav_path.name                   # e.g. SSB00050001.wav
        utt_id = wav_path.stem                  # e.g. SSB00050001
        speaker_id = wav_path.parent.name       # e.g. SSB0005

        text = utt2text.get(fname)
        if not text:
            missing_text += 1
            # 可以按需 print 出来排查
            # print(f"[WARN] 找不到文本: {fname}")
            continue

        gender = ""
        accent = ""
        if speaker_id in spk_meta:
            gender = spk_meta[speaker_id]["gender"]
            accent = spk_meta[speaker_id]["accent"]
        else:
            missing_spk_meta += 1

        rows.append({
            "audio": str(wav_path.resolve()),
            "transcription": text,
            "speaker_id": speaker_id,
            "gender": gender,
            "accent": accent,
            "utt_id": utt_id,
        })

    if not rows:
        raise RuntimeError("没有任何有效样本，请检查 content.txt 与 wav 文件名是否匹配")

    # 写出 CSV
    fieldnames = ["audio", "transcription", "speaker_id", "gender", "accent", "utt_id"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[DONE] 写出 {len(rows)} 条到 {OUT_CSV}")
    if missing_text > 0:
        print(f"[WARN] 有 {missing_text} 个 wav 在 content.txt 中没有找到文本")
    if missing_spk_meta > 0:
        print(f"[WARN] 有 {missing_spk_meta} 个说话人没有在 spk-info.txt 中找到性别/口音信息")


if __name__ == "__main__":
    main()
