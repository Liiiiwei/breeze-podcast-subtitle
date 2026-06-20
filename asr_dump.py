"""轉錄 → 逐字 word 快取（JSON）。把昂貴的 ASR 跟便宜的斷句解耦：
ASR 只跑一次寫 cache，之後改斷句規則用 build_from_cache.py 秒出 SRT。

兩種模式：
  stereo  ：連續轉完整 Stereo Mix（word_timestamps），用三軌麥能量逐字貼 spk。
  permic  ：三軌各自只轉「自己主講」的區塊（事前能量仲裁去串音），spk = MicN。

用法：
  python asr_dump.py --mode stereo --dir "<集>" --out cache_stereo.json [--limit N]
  python asr_dump.py --mode permic --dir "<集>" --out cache_permic.json [--limit N]
"""
import argparse
import glob
import json
import os

import numpy as np
import whisper

SR = 16000
FR = 0.05
LABELS = ["Mic1", "Mic2", "Mic3"]
PROMPT = ("台灣 podcast《我愛上班》訪談。主持人郝慧川、惡魔老闆岳啟儒，來賓沈奕妤 Emma。"
          "印花樂 inBlooom、Wazaiii、酷學營、茄芷袋、妮可基嫚、台灣八哥、斜槓、布料設計。")

# 串音仲裁
PREFILTER_DB = 8.0   # 區塊層級：別人比我大聲 > 此值 → 明顯串音，不轉（省時）
BLEED_MARGIN_DB = 4.0  # 句子層級：精修，丟掉殘留串音


def find_audio(folder):
    mix = None
    for pat in ("Stereo Mix*.wav", "*Stereo*Mix*.wav", "*Mix*.wav"):
        hits = sorted(glob.glob(os.path.join(folder, pat)))
        if hits:
            mix = hits[0]
            break
    raw = sorted(glob.glob(os.path.join(folder, "Track*-Mic*.wav")))
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


def speech_intervals(audio, frame=0.03, thresh_db=-50.0,
                     min_speech=0.4, merge_gap=1.0, pad=0.25):
    fl = int(frame * SR)
    n = len(audio) // fl
    if n == 0:
        return []
    frames = audio[:n * fl].reshape(n, fl)
    energy = np.sqrt((frames ** 2).mean(axis=1)) + 1e-9
    db = 20.0 * np.log10(energy)
    active = db > thresh_db
    runs, i = [], 0
    while i < n:
        if active[i]:
            j = i
            while j < n and active[j]:
                j += 1
            runs.append([i * frame, j * frame])
            i = j
        else:
            i += 1
    merged = []
    for s, e in runs:
        if merged and s - merged[-1][1] <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    total = len(audio) / SR
    out = []
    for s, e in merged:
        s = max(0.0, s - pad)
        e = min(total, e + pad)
        if e - s >= min_speech:
            out.append((s, e))
    return out


def seg_words(seg, offset=0.0, seg_id=0):
    """segment → [{w,start,end,seg,p,alp,nsp}]，無 word ts 退回整段。
    seg_id 標記來自哪個 Whisper segment（= Whisper 偵測到的氣口/句界），斷句時當硬斷點。
    p   = 逐字後驗機率（word_timestamps 才有；抓聲學模糊字的主信號）
    alp = 該 segment avg_logprob（段落品質）；nsp = no_speech_prob（近靜音幻聽）"""
    alp = seg.get("avg_logprob")
    nsp = seg.get("no_speech_prob")
    out = []
    ws = seg.get("words") or []
    if not ws:
        t = (seg.get("text") or "").strip()
        if t:
            out.append({"w": t, "start": seg["start"] + offset, "end": seg["end"] + offset,
                        "seg": seg_id, "p": None, "alp": alp, "nsp": nsp})
        return out
    for w in ws:
        t = w.get("word", "")
        if not t.strip():
            continue
        out.append({"w": t,
                    "start": w.get("start", seg["start"]) + offset,
                    "end": w.get("end", seg["end"]) + offset, "seg": seg_id,
                    "p": w.get("probability"), "alp": alp, "nsp": nsp})
    return out


def run_stereo(folder, model, limit, prompt):
    mix, raw = find_audio(folder)
    if not mix:
        raise SystemExit(f"找不到 Stereo Mix：{folder}")
    print(f"[stereo] 混音 {os.path.basename(mix)}；原始麥 {len(raw)} 軌", flush=True)
    audio = whisper.load_audio(mix)
    if limit:
        audio = audio[:int(limit * SR)]
    print(f"[stereo] 轉錄 {len(audio)/SR/60:.1f} 分（連續 + 逐字時間）…", flush=True)
    r = model.transcribe(audio, language="zh", fp16=False, initial_prompt=prompt,
                         condition_on_previous_text=True, word_timestamps=True, verbose=False)
    raw_power = [frame_power(whisper.load_audio(p)) for p in raw] if len(raw) >= 2 else []

    def label_fn(s, e):
        if not raw_power:
            return None
        return LABELS[int(np.argmax([window_db(rp, s, e) for rp in raw_power]))]

    words = []
    for sid, seg in enumerate(r["segments"]):
        for w in seg_words(seg, seg_id=sid):
            w["spk"] = label_fn(w["start"], w["end"])
            words.append(w)
    return words


def diarize(raw_power, *, speech_db=-48.0, dom_margin=3.0, smooth_frames=9,
            merge_gap=1.0, min_turn=0.40):
    """能量分軌（diarization）：逐 frame 取最大聲的麥當講者，平滑後分成 turn。
    回傳 [(spk_idx, start_s, end_s)]，互不重疊。"""
    T = min(len(p) for p in raw_power)
    P = np.stack([p[:T] for p in raw_power], axis=0)        # (n, T) 線性功率
    db = 10.0 * np.log10(P + 1e-12)                          # 每軌每frame dB
    total_db = 10.0 * np.log10(P.sum(axis=0) + 1e-12)
    active = total_db > speech_db
    dom = np.argmax(db, axis=0)                              # 最大聲的軌
    sortd = np.sort(db, axis=0)
    margin = sortd[-1] - sortd[-2]                           # 最大 vs 次大
    # 沒人說話、或最大次大太接近(同時講/串音不清) → 標 -1
    lab = np.where(active & (margin >= dom_margin), dom, -1)
    # 中位數平滑去抖
    if smooth_frames > 1:
        from collections import Counter
        sm = lab.copy()
        h = smooth_frames // 2
        for i in range(len(lab)):
            a, b = max(0, i - h), min(len(lab), i + h + 1)
            sm[i] = Counter(lab[a:b]).most_common(1)[0][0]
        lab = sm
    # 連續同講者成 turn
    turns = []
    i = 0
    while i < len(lab):
        s = lab[i]
        if s == -1:
            i += 1
            continue
        j = i
        while j < len(lab) and lab[j] == s:
            j += 1
        turns.append([int(s), i * FR, j * FR])
        i = j
    # 同講者、間隔小 → 併
    merged = []
    for spk, s, e in turns:
        if merged and merged[-1][0] == spk and s - merged[-1][2] <= merge_gap:
            merged[-1][2] = e
        else:
            merged.append([spk, s, e])
    return [(spk, s, e) for spk, s, e in merged if e - s >= min_turn]


def run_permic(folder, model, limit, prompt):
    _, raw = find_audio(folder)
    if len(raw) < 2:
        raise SystemExit(f"per-mic 需要 ≥2 軌，找到 {len(raw)}")
    raw_audio = [whisper.load_audio(p) for p in raw]
    raw_power = [frame_power(a) for a in raw_audio]

    turns = diarize(raw_power)
    if limit:
        turns = [t for t in turns if t[1] < limit]
    by_spk = {}
    for spk, s, e in turns:
        by_spk.setdefault(spk, []).append((s, e))
    total_speech = sum(e - s for _, s, e in turns)
    print(f"[permic] diarization：{len(turns)} turn，總語音 {total_speech/60:.1f} 分"
          f"（vs 全長 {len(raw_audio[0])/SR/60:.1f} 分）", flush=True)

    kept_words = []
    seg_counter = 0
    SIL = np.zeros(int(0.4 * SR), dtype=np.float32)          # turn 間插靜音，避免黏字
    for spk_idx in sorted(by_spk):
        spk = LABELS[spk_idx] if spk_idx < len(LABELS) else f"Mic{spk_idx+1}"
        segs = by_spk[spk_idx]
        # 串接該講者所有 turn 成一條連續音訊，只轉一次（省掉每段 30s padding 浪費）
        parts, table, cpos = [], [], 0.0
        for s, e in segs:
            chunk = raw_audio[spk_idx][int(s * SR):int(e * SR)]
            table.append((cpos, s, len(chunk) / SR))         # (concat起點, 原始起點, 長度)
            parts.append(chunk)
            parts.append(SIL)
            cpos += len(chunk) / SR + len(SIL) / SR
        concat = np.concatenate(parts)
        print(f"[permic] {spk}: {len(segs)} turn → 串接 {len(concat)/SR/60:.1f} 分，轉錄中…", flush=True)
        r = model.transcribe(concat, language="zh", fp16=False, initial_prompt=prompt,
                             condition_on_previous_text=True, word_timestamps=True, verbose=False)

        def map_time(tc, table=table):
            for cstart, ostart, dur in table:
                if cstart <= tc <= cstart + dur + 1e-3:
                    return ostart + (tc - cstart)
            best = table[0][1]
            for cstart, ostart, dur in table:
                if tc >= cstart:
                    best = ostart + min(tc - cstart, dur)
            return best

        for seg in r["segments"]:
            txt = (seg.get("text") or "").strip()
            if not txt:
                continue
            # 過濾噪音幻聽：低信心 or 近靜音
            if seg.get("no_speech_prob", 0) > 0.5 or seg.get("avg_logprob", -2) < -0.9:
                continue
            for w in seg_words(seg, seg_id=seg_counter):
                w["start"] = map_time(w["start"])
                w["end"] = map_time(w["end"])
                w["spk"] = spk
                kept_words.append(w)
            seg_counter += 1

    kept_words.sort(key=lambda w: w["start"])
    return kept_words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["stereo", "permic"], required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=float, default=None)
    ap.add_argument("--prompt", default=PROMPT)
    args = ap.parse_args()

    folder = os.path.abspath(args.dir)
    print("載入 Breeze-ASR-25（CPU）…", flush=True)
    model = whisper.load_model("breeze-asr-25", device="cpu")

    if args.mode == "stereo":
        words = run_stereo(folder, model, args.limit, args.prompt)
    else:
        words = run_permic(folder, model, args.limit, args.prompt)

    payload = {"mode": args.mode, "dir": folder, "limit": args.limit,
               "prompt": args.prompt, "n_words": len(words), "words": words}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"✅ 快取 {len(words)} 字 → {args.out}", flush=True)


if __name__ == "__main__":
    main()
