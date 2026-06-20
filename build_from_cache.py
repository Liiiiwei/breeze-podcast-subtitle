"""讀 asr_dump.py 的 word 快取 → 套節奏式斷句 → 出 SRT（無標 + 含講者）。
便宜可反覆跑，用來調斷句參數而不必重轉錄。

用法：
  python build_from_cache.py cache.json out_plain.srt [out_labeled.srt]
        [--max-w 16] [--gap 0.40] [--particle-gap 0.18] [--corrections corrections_syy.json]
"""
import argparse
import json

from rhythm_segment import split_words_to_cues, smooth_spk


def fmt_ts(t):
    if t < 0:
        t = 0
    h = int(t // 3600); m = int((t % 3600) // 60); s = int(t % 60)
    ms = int(round((t - int(t)) * 1000))
    if ms == 1000:
        ms = 0; s += 1
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def load_corrections(path):
    """只做『多字唯一』可安全硬替換的詞（names）；style/context 不在此做。"""
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    repl = {}
    for canon, variants in (data.get("names") or {}).items():
        for v in variants:
            repl[v] = canon
    return repl


def apply_corrections(text, repl):
    for wrong, right in repl.items():
        if wrong in text:
            text = text.replace(wrong, right)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("out_plain")
    ap.add_argument("out_labeled", nargs="?")
    ap.add_argument("--max-w", type=float, default=16.0)
    ap.add_argument("--gap", type=float, default=0.40)
    ap.add_argument("--particle-gap", type=float, default=0.18)
    ap.add_argument("--smooth", type=float, default=0.6, help="stereo 麥能量時間域濾波視窗(秒)")
    ap.add_argument("--corrections", default=None)
    args = ap.parse_args()

    with open(args.cache, encoding="utf-8") as f:
        payload = json.load(f)
    words = payload["words"]
    mode = payload.get("mode", "stereo")
    repl = load_corrections(args.corrections)

    # stereo：逐字麥能量會抖動且會切斷詞 → 純停頓+字數上限斷句(乾淨、不切詞)，
    #         每卡再以麥能量多數貼標(盡力)。無停頓的 turn 會併卡共用一標 = stereo 先天限制。
    # permic：每軌就是一個人 → spk 可靠，用 spk 變更當斷點(精準 turn)。
    if mode == "permic":
        spk_break, relabel = True, False
    else:
        spk_break, relabel = False, True

    cues = split_words_to_cues(words, max_w=args.max_w, gap_break=args.gap,
                               particle_gap=args.particle_gap,
                               spk_break=spk_break, relabel=relabel)
    if repl:
        cues = [(s, e, spk, apply_corrections(t, repl)) for (s, e, spk, t) in cues]

    has_spk = any(spk for _, _, spk, _ in cues)
    writes = [(args.out_plain, False)]
    if args.out_labeled and has_spk:
        writes.append((args.out_labeled, True))
    for path, labeled in writes:
        with open(path, "w", encoding="utf-8") as f:
            for i, (s, e, spk, txt) in enumerate(cues, 1):
                if e <= s:
                    e = s + 0.4
                line = f"[{spk}] {txt}" if (labeled and spk) else txt
                f.write(f"{i}\n{fmt_ts(s)} --> {fmt_ts(e)}\n{line}\n\n")
        print(f"✅ {path}  ({len(cues)} 卡)")


if __name__ == "__main__":
    main()
