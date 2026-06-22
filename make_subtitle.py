"""
通用 Podcast 字幕產生器(不漏抓版)。

對任一集資料夾:
  1. 自動找到 Stereo Mix 與 Track*-Mic*.wav
  2. 連續轉錄完整混音(不切片、不閘門)→ 不漏內容
  3. initial_prompt 餵主持人/來賓/詞表 → 名字一開始就對
  4. 用原始麥能量逐句貼講者標籤
  5. 輸出「無標 / 含講者」兩版 SRT(以資料夾名命名)

用法:
  python make_subtitle.py --dir "/path/to/20260508 沈奕妤" --guest "沈奕妤 Emma"
  python make_subtitle.py --dir "..." --guest "某某" --terms "歐舒丹、插畫、品牌設計"
選項:
  --hosts   主持人(預設:郝慧川、惡魔老闆岳啟儒)
  --limit   只轉前 N 秒(測試用)
"""
import argparse
import glob
import os
import re

import numpy as np
import whisper

from srt_segment import balanced_split, char_width, add_words
from rhythm_segment import segments_to_words, split_words_to_cues, smooth_spk

SR = 16000
HOSTS_DEFAULT = "郝慧川、惡魔老闆岳啟儒"
FR = 0.05


def find_audio(folder):
    # 音檔可能在集根目錄，或 podcast-toolkit 慣例的 01_母帶 / 02_素材 子資料夾。
    # 依序找（根目錄優先，向後相容），找到就用該層。
    search_dirs = [
        folder,
        os.path.join(folder, "01_母帶"),
        os.path.join(folder, "02_素材"),
    ]
    mix = None
    for d in search_dirs:
        for pat in ("Stereo Mix*.wav", "*Stereo*Mix*.wav", "*Mix*.wav"):
            hits = sorted(glob.glob(os.path.join(d, pat)))
            if hits:
                mix = hits[0]
                break
        if mix:
            break
    raw = []
    for d in search_dirs:
        raw = sorted(glob.glob(os.path.join(d, "Track*-Mic*.wav")))
        if raw:
            break
    return mix, raw


def frame_power(audio):
    fl = int(FR * SR)
    n = len(audio) // fl
    if n == 0:
        return np.zeros(0)
    f = audio[:n * fl].reshape(n, fl)
    return (f.astype(np.float64) ** 2).mean(axis=1)


def window_db(power, s, e):
    fi = int(s / FR)
    fj = max(fi + 1, int(e / FR))
    seg = power[fi:fj]
    if len(seg) == 0:
        return -99.0
    return 10.0 * np.log10(float(seg.mean()) + 1e-12)


def fmt_ts(t):
    if t < 0:
        t = 0
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms == 1000:
        ms = 0; s += 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _seg_to_chars(seg):
    """把 segment 的詞級時間戳展成「每字 (字, start, end)」;無詞時間戳則線性插值。"""
    words = seg.get("words") or []
    chars = []
    for w in words:
        t = w.get("word", "")
        s = w.get("start"); e = w.get("end")
        if s is None:
            s = seg["start"]
        if e is None:
            e = seg["end"]
        if not t:
            continue
        d = (e - s) / len(t)
        for i, c in enumerate(t):
            chars.append((c, s + d * i, s + d * (i + 1)))
    if not chars:
        text = seg["text"]
        s, e = seg["start"], seg["end"]
        d = (e - s) / max(1, len(text))
        chars = [(c, s + d * i, s + d * (i + 1)) for i, c in enumerate(text)]
    return chars


def split_segment(seg, max_w):
    """把一個 segment 依「平衡 + 邊界對齊」規則拆成多個 (start, end, text)。"""
    text = seg["text"].strip()
    if char_width(text) <= max_w:
        return [(seg["start"], seg["end"], text)]
    chars = _seg_to_chars(seg)
    out = []
    for a, b in balanced_split(chars, max_w):
        piece = re.sub(r"\s+", " ", "".join(c[0] for c in chars[a:b])).strip()
        if piece:
            out.append((chars[a][1], chars[b - 1][2], piece))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="該集資料夾")
    ap.add_argument("--guest", default="", help="來賓姓名(餵 initial_prompt)")
    ap.add_argument("--terms", default="", help="額外專有名詞,逗號分隔")
    ap.add_argument("--hosts", default=HOSTS_DEFAULT)
    ap.add_argument("--limit", type=float, default=None)
    ap.add_argument("--max-line", type=float, default=16.0, help="每句最長全形字數,過長自動拆")
    ap.add_argument("--quiet", action="store_true", help="關閉進度條(給 UI 用)")
    args = ap.parse_args()

    folder = os.path.abspath(args.dir)
    name = os.path.basename(folder.rstrip(os.sep)) or "output"
    mix, raw = find_audio(folder)
    if not mix:
        raise SystemExit(f"❌ 在 {folder} 找不到 Stereo Mix 音檔")
    print(f"混音: {os.path.basename(mix)}")
    print(f"原始麥: {len(raw)} 軌" + (" → 會貼講者標籤" if len(raw) >= 2 else " → 無法貼標"))

    prompt = f"台灣 podcast《我愛上班》訪談。主持人{args.hosts}。"
    if args.guest:
        prompt += f"來賓{args.guest}。"
    if args.terms:
        prompt += args.terms.replace(",", "、")
    print(f"提示詞: {prompt}")

    # 把來賓姓名/額外詞彙加進 jieba 詞庫,避免斷行時被切開
    extra = args.guest.split() + re.split(r"[,、]", args.terms)
    add_words(extra)

    print("載入模型(CPU)…", flush=True)
    model = whisper.load_model("breeze-asr-25", device="cpu")

    print("載入完整混音…", flush=True)
    audio = whisper.load_audio(mix)
    if args.limit:
        audio = audio[:int(args.limit * SR)]

    print(f"轉錄中(連續模式,詞級時間戳,{len(audio)/SR/60:.1f} 分)… 這段較久,請耐心等候", flush=True)
    result = model.transcribe(
        audio, language="zh", fp16=False,
        initial_prompt=prompt, condition_on_previous_text=True,
        word_timestamps=True,
        verbose=(None if args.quiet else False),
    )
    # 先載入原始三軌,讓斷句時就能逐字判斷講者
    labels = [f"Mic{i+1}" for i in range(len(raw))]
    raw_power = []
    if len(raw) >= 2:
        print("載入原始三軌(貼標用)…", flush=True)
        raw_power = [frame_power(whisper.load_audio(p)) for p in raw]

    def label_fn(s, e):
        if not raw_power:
            return None
        return labels[int(np.argmax([window_db(rp, s, e) for rp in raw_power]))]

    # 節奏式斷句:照 Whisper 氣口/句界 + 停頓 + 語氣詞斷,接近人工 Fumo 版的節奏
    # (比純 16 字平衡切更接近人工:卡更短、邊界落在語氣轉折上)。
    # stereo 混音逐字麥能量會抖動 → 先時間域多數濾波,每卡再取多數講者(不切斷詞)。
    words = segments_to_words(result["segments"], label_fn=label_fn)
    if raw_power:
        words = smooth_spk(words, win=0.6)
    rows = split_words_to_cues(words, max_w=args.max_line,
                               spk_break=False, relabel=bool(raw_power))
    print(f"轉錄 + 節奏式斷句後共 {len(rows)} 句(每句 ≤ {args.max_line:g} 全形字)", flush=True)

    out_plain = os.path.join(folder, f"{name}_字幕.srt")
    writes = [(out_plain, False)]
    if raw_power:
        writes.append((os.path.join(folder, f"{name}_字幕_含講者.srt"), True))

    for path, labeled in writes:
        with open(path, "w", encoding="utf-8") as f:
            for i, (s, e, spk, txt) in enumerate(rows, 1):
                if e <= s:
                    e = s + 0.5
                line = f"[{spk}] {txt}" if (labeled and spk) else txt
                f.write(f"{i}\n{fmt_ts(s)} --> {fmt_ts(e)}\n{line}\n\n")
        print(f"✅ {path}")


if __name__ == "__main__":
    main()
