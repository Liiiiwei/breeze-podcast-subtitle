"""節奏式斷句（rhythm segmentation）。

對齊人工 Fumo 版的觀察：人工斷句不是照「固定字數」切，而是照「說話節奏」切——
  1) 講者換人 → 一定換卡
  2) 氣口/停頓（詞與詞之間的空檔）→ 換卡
  3) 句末語氣詞（的/了/嗎/啊/喔…）後若有小停頓 → 傾向換卡
  4) 以上都沒有、卡太長 → 才退回 balanced_split 的 ≤16 全形字硬上限

結果比純 balanced_split 更接近人工：卡數更多、每卡更短、邊界落在語氣轉折上。
輸入是「逐字時間戳」（Breeze word_timestamps=True 會給），所以能看到停頓。
"""
import re

from srt_segment import balanced_split, char_width

# 句末語氣詞 / 標點（其後是好的斷點）
_PARTICLE_END = set("的了嗎呢吧啊喔噢啦嘛呀耶欸哦囉嘍，。、！？；：")


def _run_to_chars(words):
    """一串 word dict → [(字, start, end)]，詞內以時間線性插值到每字。"""
    chars = []
    for w in words:
        t = (w.get("w") or "").strip()
        if not t:
            continue
        s = w.get("start")
        e = w.get("end")
        if s is None or e is None or e <= s:
            # 沒有可信時間 → 退回 0 長度，後面用相鄰補
            s = s if s is not None else (chars[-1][2] if chars else 0.0)
            e = max(s, e if e is not None else s)
        d = (e - s) / len(t)
        for i, c in enumerate(t):
            chars.append((c, s + d * i, s + d * (i + 1)))
    return chars


def smooth_spk(words, *, win=0.6):
    """時間域多數濾波，消掉逐字麥能量的單字跳動（stereo 用）。
    每字的 spk 改成「以它為中心 ±win 秒內所有字」的時長加權多數。
    用原始標籤計算（不串接），real 的長 turn 不受影響，孤立翻轉被吸收。"""
    from collections import Counter
    centers = [((w.get("start") or 0) + (w.get("end") or 0)) / 2 for w in words]
    out = []
    n = len(words)
    for i, w in enumerate(words):
        if w.get("spk") is None:
            out.append(dict(w))
            continue
        c = Counter()
        j = i
        while j >= 0 and centers[i] - centers[j] <= win:
            sp = words[j].get("spk")
            if sp is not None:
                c[sp] += max((words[j].get("end") or 0) - (words[j].get("start") or 0), 0.01)
            j -= 1
        j = i + 1
        while j < n and centers[j] - centers[i] <= win:
            sp = words[j].get("spk")
            if sp is not None:
                c[sp] += max((words[j].get("end") or 0) - (words[j].get("start") or 0), 0.01)
            j += 1
        nw = dict(w)
        nw["spk"] = c.most_common(1)[0][0] if c else w.get("spk")
        out.append(nw)
    return out


def _majority_spk(run):
    """run 內各字的 spk 取多數（以時長加權），用於 stereo 麥能量貼標去抖動。"""
    from collections import Counter
    c = Counter()
    for w in run:
        spk = w.get("spk")
        if spk is None:
            continue
        dur = (w.get("end") or 0) - (w.get("start") or 0)
        c[spk] += max(dur, 0.01)
    return c.most_common(1)[0][0] if c else (run[0].get("spk") if run else None)


def split_words_to_cues(words, *, max_w=16.0, gap_break=0.40,
                        particle_gap=0.18, spk_break=True, relabel=False):
    """words: [{'w','start','end','spk'}]（時間秒，spk 可為 None）。
    回傳 [(start, end, spk, text)]。

    斷點規則（硬斷成 run，再對每個 run 套 balanced_split ≤max_w）：
      - 講者變更（spk_break=True 時；per-mic 用，spk 可靠）
      - 與前一字的停頓 >= gap_break
      - 前一字是語氣詞/標點，且停頓 >= particle_gap（語氣轉折的小氣口）

    spk_break=False：忽略 spk 變更當斷點（stereo 用，逐字麥能量會抖動）。
    relabel=True：每個 cue 的 spk 改取 run 內多數（stereo 用，去抖動）。
    """
    # 1) 先把 words 切成 runs
    runs = []
    cur = []
    prev_end = None
    prev_spk = None
    prev_seg = None
    prev_tail = ""
    for w in words:
        txt = (w.get("w") or "").strip()
        if not txt:
            continue
        spk = w.get("spk")
        seg = w.get("seg")
        st = w.get("start")
        gap = (st - prev_end) if (st is not None and prev_end is not None) else 0.0
        brk = False
        if cur:
            if seg is not None and prev_seg is not None and seg != prev_seg:
                brk = True   # Whisper segment 邊界 = 偵測到的氣口/句界
            elif spk_break and spk != prev_spk:
                brk = True
            elif gap >= gap_break:
                brk = True
            elif prev_tail and prev_tail[-1] in _PARTICLE_END and gap >= particle_gap:
                brk = True
        if brk:
            runs.append(cur)
            cur = []
        cur.append(w)
        prev_end = w.get("end") if w.get("end") is not None else st
        prev_spk = spk
        prev_seg = seg
        prev_tail = txt
    if cur:
        runs.append(cur)

    # 2) 每個 run 套 balanced_split 上限切，產出 cue
    out = []
    for run in runs:
        chars = _run_to_chars(run)
        if not chars:
            continue
        spk = _majority_spk(run) if relabel else run[0].get("spk")
        total = sum(char_width(c[0]) for c in chars)
        if total <= max_w:
            pieces = [(0, len(chars))]
        else:
            pieces = balanced_split(chars, max_w)
        for a, b in pieces:
            piece = re.sub(r"\s+", " ", "".join(c[0] for c in chars[a:b])).strip()
            if piece:
                out.append((chars[a][1], chars[b - 1][2], spk, piece))
    return out


def segments_to_words(segments, *, label_fn=None):
    """Whisper result['segments'] → 扁平 word list（含 spk）。
    label_fn(start, end) -> spk 字串（可選，stereo 模式用麥能量貼標）。
    無 word_timestamps 的 segment 退回整段當一個 word。"""
    words = []
    for seg in segments:
        segwords = seg.get("words") or []
        if not segwords:
            txt = (seg.get("text") or "").strip()
            if txt:
                s, e = seg["start"], seg["end"]
                words.append({"w": txt, "start": s, "end": e,
                              "spk": label_fn(s, e) if label_fn else None})
            continue
        for w in segwords:
            t = w.get("word", "")
            if not t.strip():
                continue
            s = w.get("start", seg["start"])
            e = w.get("end", seg["end"])
            words.append({"w": t, "start": s, "end": e,
                          "spk": label_fn(s, e) if label_fn else None})
    return words
