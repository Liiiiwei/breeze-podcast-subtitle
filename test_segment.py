"""斷句引擎測試:評分修正(srt_segment)+ run 斷點詞界校正 snap(rhythm_segment)
+ word 快取 end-to-end(build_from_cache)。

直跑:.venv/bin/python test_segment.py
不依賴 pytest,全部用 assert;任何一項失敗即非零退出。
"""
import json
import os
import subprocess
import sys
import tempfile

from srt_segment import balanced_split, DANGLE_TAIL2
from rhythm_segment import split_words_to_cues

HERE = os.path.dirname(os.path.abspath(__file__))
PASS = []


def ok(name, cond, detail=""):
    assert cond, f"❌ {name}: {detail}"
    PASS.append(name)
    print(f"  ✓ {name}" + (f"  ({detail})" if detail else ""))


def mk_words(text, *, dur=0.2, seg_at=None, spk_at=None, gap_at=None, gap=0.0):
    """把字串展成逐字 word list(模擬 whisper 常把雙字詞拆成單字 word)。
    seg_at:  index i 起 seg 從 0 變 1(模擬 whisper segment 邊界落在 i 前)
    spk_at:  index i 起講者從 Mic1 變 Mic2(None = 全 None,不貼標)
    gap_at:  index i 前插入 gap 秒的停頓
    """
    words, t = [], 0.0
    for i, c in enumerate(text):
        if gap_at is not None and i == gap_at:
            t += gap
        w = {"w": c, "start": round(t, 3), "end": round(t + dur, 3),
             "seg": (0 if (seg_at is None or i < seg_at) else 1)}
        if spk_at is not None:
            w["spk"] = "Mic1" if i < spk_at else "Mic2"
        else:
            w["spk"] = None
        words.append(w)
        t += dur
    return words


def split_text(s, max_w=16.0):
    chars = [(c, i, i + 1) for i, c in enumerate(s)]
    return ["".join(s[a:b]) for a, b in balanced_split(chars, max_w)]


# ---------- (a) seg 邊界落在「耳機」中間、gap 小 → snap ----------
def test_a():
    words = mk_words("我們都會戴耳機聽節目", seg_at=6)   # 邊界落在 耳|機
    cues = split_words_to_cues(words, spk_break=False, relabel=False)
    texts = [t for _, _, _, t in cues]
    ok("a. seg邊界切「耳機」被 snap", texts == ["我們都會戴", "耳機聽節目"],
       " | ".join(texts))
    # 時間戳跟著字走:「耳」(原 A 尾)的 start 要成為第二卡的 start
    ok("a. 搬字後時間戳跟著走", abs(cues[1][0] - 1.0) < 1e-6,
       f"第二卡 start={cues[1][0]}")


# ---------- (a2) seg 邊界 gap 大(真氣口)→ 不 snap ----------
def test_a2():
    words = mk_words("我們都會戴耳機聽節目", seg_at=6, gap_at=6, gap=0.5)  # 0.5 ≥ 0.35
    cues = split_words_to_cues(words, spk_break=False, relabel=False)
    texts = [t for _, _, _, t in cues]
    ok("a2. seg邊界 gap≥0.35 保留氣口", texts == ["我們都會戴耳", "機聽節目"],
       " | ".join(texts))


# ---------- (b) spk 斷點落在「因為」中間、gap 小 → snap 且講者採目標 run ----------
def test_b():
    # jieba(dict.txt.big)切:你問/我/為/什麼… → 用「因為」當跨界詞(確認是一個詞)
    words = mk_words("你怎麼會知道因為我在現場", spk_at=7)   # 換人落在 因|為
    cues = split_words_to_cues(words, spk_break=True, relabel=False)
    texts = [t for _, _, _, t in cues]
    spks = [s for _, _, s, _ in cues]
    ok("b. spk斷切「因為」被 snap", texts == ["你怎麼會知道", "因為我在現場"],
       " | ".join(texts))
    ok("b. 搬過去的「因」講者改採目標 run", spks == ["Mic1", "Mic2"], str(spks))


# ---------- (b2) spk 斷點 gap 大(0.7 ≥ 0.6)→ 不 snap ----------
def test_b2():
    words = mk_words("你怎麼會知道因為我在現場", spk_at=7, gap_at=7, gap=0.7)
    cues = split_words_to_cues(words, spk_break=True, relabel=False)
    texts = [t for _, _, _, t in cues]
    ok("b2. spk斷 gap≥0.6 不 snap", texts == ["你怎麼會知道因", "為我在現場"],
       " | ".join(texts))


# ---------- (b3) gap 斷點(≥0.8)本身就是真停頓 → 永不 snap ----------
def test_b3():
    words = mk_words("我們都會戴耳機聽節目", gap_at=6, gap=0.9)   # 停頓落在 耳|機
    cues = split_words_to_cues(words, spk_break=False, relabel=False)
    texts = [t for _, _, _, t in cues]
    ok("b3. gap斷(≥0.8)不 snap", texts == ["我們都會戴耳", "機聽節目"],
       " | ".join(texts))


# ---------- (c) 「太吵的時候」不再切成「太吵的|時候」 ----------
def test_c():
    s = "因為現場真的太吵的時候你根本聽不見耳機裡的聲音"
    pieces = split_text(s)
    ok("c. 定語「的」不再被加分切開", "太吵的|時候" not in "|".join(pieces),
       " | ".join(pieces))


# ---------- (d) 「來自於OB車上」不切在「於」後 ----------
def test_d():
    s = "這個訊號其實是來自於OB車上的導播設備傳過來的"
    pieces = split_text(s)
    ok("d. 「於」不當卡尾", not any(p.endswith("於") for p in pieces[:-1]),
       " | ".join(pieces))


# ---------- (e) 卡尾不再掛連接詞(然後/甚至/或是…) ----------
def test_e():
    cases = [
        "那一場的觀眾少說有五百人甚至可能快要一千人了",    # 舊評分:「…甚至|可能…」
        "他們家的隔音跟通風設備真的做得很不錯然後我們就決定租下來了",
        "他們平常會用無線的麥克風或是直接拉一條線到控台",
    ]
    for s in cases:
        pieces = split_text(s)
        ok(f"e. 無掛尾連接詞({s[:8]}…)",
           not any(p[-2:] in DANGLE_TAIL2 for p in pieces[:-1]),
           " | ".join(pieces))


# ---------- (f) 假快取 end-to-end:build_from_cache 讀 10 字快取出 SRT ----------
def test_f():
    words = mk_words("我們都會戴耳機聽節目", seg_at=6)   # 10 個 word,邊界切在 耳|機
    for w in words:
        w["spk"] = "Mic1"
        w["p"] = 0.9
        w["alp"] = -0.2
        w["nsp"] = 0.01
    payload = {"mode": "stereo", "dir": "/tmp/fake", "limit": None,
               "prompt": "測試", "n_words": len(words), "words": words}
    with tempfile.TemporaryDirectory() as td:
        cache = os.path.join(td, "cache.json")
        out = os.path.join(td, "out.srt")
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        r = subprocess.run([sys.executable, os.path.join(HERE, "build_from_cache.py"),
                            cache, out], capture_output=True, text=True)
        ok("f. build_from_cache 跑通", r.returncode == 0, r.stderr.strip()[-200:])
        with open(out, encoding="utf-8") as f:
            srt = f.read()
        lines = [ln for ln in srt.splitlines() if ln and "-->" not in ln
                 and not ln.isdigit()]
        ok("f. 快取出的 SRT 卡文含 snap 修正", lines == ["我們都會戴", "耳機聽節目"],
           " | ".join(lines))


if __name__ == "__main__":
    for fn in [test_a, test_a2, test_b, test_b2, test_b3,
               test_c, test_d, test_e, test_f]:
        print(fn.__doc__ or fn.__name__)
        fn()
    print(f"\n✅ 全部通過({len(PASS)} 項斷言)")
