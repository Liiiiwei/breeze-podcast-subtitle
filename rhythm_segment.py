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

from srt_segment import balanced_split, char_width, word_break_ok

# 句末語氣詞 / 標點（其後是好的斷點）
_PARTICLE_END = set("的了嗎呢吧啊喔噢啦嘛呀耶欸哦囉嘍，。、！？；：")

# run 硬斷點的詞界校正（snap）門檻：斷點兩側 gap 小於門檻＝不是真氣口，才校正。
# gap 大代表真停頓（氣口），切在那裡是對的，不動。
SEG_SNAP_GAP = 0.35   # whisper seg 邊界斷／語氣詞斷（word 邊界常把雙字詞拆成兩個 word）
SPK_SNAP_GAP = 0.60   # 講者變更斷（stereo 麥能量貼標本來就抖，放寬）


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


def stabilize_spk(words, *, min_run=0.4):
    """消除過短的講者-run（重疊區逐字麥能量抖動造成的單字翻轉）。

    反覆把「總時長 < min_run 秒」的講者-run 併入較長的相鄰 run（沿用其 spk），
    避免 spk_break=True 在三人搶話的抖動區把字切碎（如「我/看/你」各成一卡）。
    乾淨輪流講話的真實換人（turn ≥ min_run）不受影響，仍正常切卡。
    None spk（無講者標、單軌集）整串原樣回傳。"""
    if not any(w.get("spk") is not None for w in words):
        return [dict(w) for w in words]
    out = [dict(w) for w in words]

    def _dur(w):
        return max((w.get("end") or 0) - (w.get("start") or 0), 0.0)

    while True:
        # 依 spk 切連續 run：[i0, i1, spk, 總時長]
        runs = []
        for i, w in enumerate(out):
            sp = w.get("spk")
            if runs and runs[-1][2] == sp:
                runs[-1][1] = i
                runs[-1][3] += _dur(w)
            else:
                runs.append([i, i, sp, _dur(w)])
        # 找最短、有講者、短於門檻、且有可併鄰居的 run
        target = None
        for idx, r in enumerate(runs):
            if r[2] is None or r[3] >= min_run:
                continue
            left = runs[idx - 1] if idx > 0 else None
            right = runs[idx + 1] if idx < len(runs) - 1 else None
            if not any(x and x[2] is not None for x in (left, right)):
                continue  # 兩側都無有效講者 → 跳過，避免卡死
            if target is None or r[3] < runs[target][3]:
                target = idx
        if target is None:
            break
        # 併入時長較長的鄰居（沿用其 spk）
        left = runs[target - 1] if target > 0 else None
        right = runs[target + 1] if target < len(runs) - 1 else None
        best = max((r for r in (left, right) if r and r[2] is not None),
                   key=lambda r: r[3])
        for k in range(runs[target][0], runs[target][1] + 1):
            out[k]["spk"] = best[2]
    return out


def _snap_runs_to_word_boundary(runs, reasons, *, seg_snap_gap=SEG_SNAP_GAP,
                                spk_snap_gap=SPK_SNAP_GAP):
    """把 run 硬斷點校正到 jieba 詞界，修「跨卡切詞」（然/後、耳/機、尤/其是…）。

    runs：word-dict list 的 list；reasons[i]：runs[i] 與 runs[i+1] 間的斷點成因
    （'seg'／'spk'／'gap'／'particle'）。

    規則：
      - 只在斷點兩側時間 gap < 門檻時才動（gap 大＝真氣口，保留原斷點）：
        seg/particle 斷用 seg_snap_gap；spk 斷用 spk_snap_gap；gap 斷（本身 ≥ gap_break）永不校正。
      - 兩個 run 的文字接起來跑 jieba，若有 ≥2 字的詞跨越斷點 →
        把斷點移到該詞的起點或終點：移動字數較少的那側；平手 → 整個詞歸後段。
      - 只搬「整個 word tuple」（時間戳跟著字走）；湊不出剛好的字數（whisper 給了
        跨詞界的多字 word）→ 放棄該處校正。搬過去的字，講者標籤改採目標 run 的講者。
    """
    thr_map = {"seg": seg_snap_gap, "particle": seg_snap_gap, "spk": spk_snap_gap}
    for i in range(len(runs) - 1):
        a, b = runs[i], runs[i + 1]
        thr = thr_map.get(reasons[i]) if i < len(reasons) else None
        if thr is None or not a or not b:
            continue
        a_end = a[-1].get("end")
        b_start = b[0].get("start")
        gap = (b_start - a_end) if (a_end is not None and b_start is not None) else 0.0
        if gap >= thr:
            continue                                  # 真氣口 → 保留
        a_text = "".join((w.get("w") or "").strip() for w in a)
        b_text = "".join((w.get("w") or "").strip() for w in b)
        pts = word_break_ok(a_text + b_text)
        pos = len(a_text)
        if pts is None or pos in pts:
            continue                                  # 沒 jieba／本來就在詞界 → 不動
        prev_pt = max(p for p in pts if p < pos)      # 跨界詞的起點
        next_pt = min(p for p in pts if p > pos)      # 跨界詞的終點
        m_left = pos - prev_pt                        # 從 A 尾搬 m_left 字去 B（整詞歸後段）
        m_right = next_pt - pos                       # 從 B 頭搬 m_right 字來 A（整詞歸前段）

        def _take(words_, m, from_tail):
            """從 run 尾／頭湊「剛好 m 個字」的整批 word tuple；湊不齊或會搬空回 None。"""
            got, cnt = [], 0
            src = reversed(words_) if from_tail else iter(words_)
            for w in src:
                if cnt >= m:
                    break
                got.append(w)
                cnt += len((w.get("w") or "").strip())
            if cnt != m or len(got) >= len(words_):
                return None
            return list(reversed(got)) if from_tail else got

        # 移動字數較少的那側優先；平手 → 整個詞歸後段（A 尾搬去 B）
        plans = sorted([("left", m_left), ("right", m_right)],
                       key=lambda x: (x[1], 0 if x[0] == "left" else 1))
        for side, m in plans:
            if side == "left":
                mv = _take(a, m, from_tail=True)
                if mv is None:
                    continue
                tgt_spk = b[0].get("spk")
                mv = [dict(w, spk=tgt_spk) for w in mv]
                runs[i] = a[:len(a) - len(mv)]
                runs[i + 1] = mv + b
            else:
                mv = _take(b, m, from_tail=False)
                if mv is None:
                    continue
                tgt_spk = a[-1].get("spk")
                mv = [dict(w, spk=tgt_spk) for w in mv]
                runs[i] = a + mv
                runs[i + 1] = b[len(mv):]
            break
    return [r for r in runs if r]


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


def split_words_to_cues(words, *, max_w=16.0, gap_break=0.80,
                        particle_gap=0.60, spk_break=True, relabel=False,
                        snap=True, seg_snap_gap=SEG_SNAP_GAP,
                        spk_snap_gap=SPK_SNAP_GAP):
    """words: [{'w','start','end','spk'}]（時間秒，spk 可為 None）。
    回傳 [(start, end, spk, text)]。

    斷點規則（硬斷成 run，再對每個 run 套 balanced_split ≤max_w）：
      - 講者變更（spk_break=True 時；per-mic 用，spk 可靠）
      - 與前一字的停頓 >= gap_break
      - 前一字是語氣詞/標點，且停頓 >= particle_gap（語氣轉折的小氣口）

    spk_break=False：忽略 spk 變更當斷點（stereo 用，逐字麥能量會抖動）。
    relabel=True：每個 cue 的 spk 改取 run 內多數（stereo 用，去抖動）。
    snap=True：run 硬斷點落在 jieba 詞中間、且兩側 gap 小（非真氣口）時，
    把斷點校正到詞界（見 _snap_runs_to_word_boundary），修「跨卡切詞」。
    """
    # 1) 先把 words 切成 runs（並記下每個斷點的成因，供詞界校正判斷門檻）
    runs = []
    reasons = []              # reasons[i] = runs[i] 與 runs[i+1] 之間的斷點成因
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
        reason = None
        if cur:
            if seg is not None and prev_seg is not None and seg != prev_seg:
                reason = "seg"       # Whisper segment 邊界 = 偵測到的氣口/句界
            elif spk_break and spk != prev_spk:
                reason = "spk"
            elif gap >= gap_break:
                reason = "gap"
            elif prev_tail and prev_tail[-1] in _PARTICLE_END and gap >= particle_gap:
                reason = "particle"
        if reason:
            runs.append(cur)
            reasons.append(reason)
            cur = []
        cur.append(w)
        prev_end = w.get("end") if w.get("end") is not None else st
        prev_spk = spk
        prev_seg = seg
        prev_tail = txt
    if cur:
        runs.append(cur)

    # 1.5) 詞界校正：run 內容定案後才切卡（要在 balanced_split 之前做）
    if snap and len(runs) > 1:
        runs = _snap_runs_to_word_boundary(runs, reasons,
                                           seg_snap_gap=seg_snap_gap,
                                           spk_snap_gap=spk_snap_gap)

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
    無 word_timestamps 的 segment 退回整段當一個 word。

    欄位 schema 對齊 asr_dump.py 的 seg_words()：
      seg = 來源 segment 序號（Whisper 氣口/句界，斷句時當硬斷點）
      p   = 逐字後驗機率；alp = segment avg_logprob；nsp = no_speech_prob
    這樣 make_subtitle.py dump 的 word 快取可直接餵 build_from_cache.py 迭代斷句。"""
    words = []
    for sid, seg in enumerate(segments):
        alp = seg.get("avg_logprob")
        nsp = seg.get("no_speech_prob")
        segwords = seg.get("words") or []
        if not segwords:
            txt = (seg.get("text") or "").strip()
            if txt:
                s, e = seg["start"], seg["end"]
                words.append({"w": txt, "start": s, "end": e, "seg": sid,
                              "p": None, "alp": alp, "nsp": nsp,
                              "spk": label_fn(s, e) if label_fn else None})
            continue
        for w in segwords:
            t = w.get("word", "")
            if not t.strip():
                continue
            s = w.get("start", seg["start"])
            e = w.get("end", seg["end"])
            words.append({"w": t, "start": s, "end": e, "seg": sid,
                          "p": w.get("probability"), "alp": alp, "nsp": nsp,
                          "spk": label_fn(s, e) if label_fn else None})
    return words
