"""字幕斷句:中文斷詞(jieba)+ 平衡切分 + 不切斷詞 + 不留孤字。
供 make_subtitle.py 與 resplit_srt.py 共用。"""
import math
import os

PUNCT = "，。、！？；：,!?;:"
# 適合「切在其後」的字(真句末語氣詞/標點)。
# 注意:「的」已移除——定語的「的」(太吵的|時候)切開會拆散修飾語與中心語;
# 真句末的「…是很有興趣的」不靠加分也切得到(其後常伴隨停頓/句界)。
PARTICLE_END = set("了嗎呢吧啊喔噢啦嘛呀耶欸哦囉嘍" + PUNCT)
# 適合「切在其前」的連接詞/轉折(2 字)
CONJ2 = {"然後", "可是", "但是", "所以", "因為", "就是", "而且", "不過",
         "另外", "其實", "譬如", "比如", "當然", "可能", "如果", "那個", "這個", "後來"}
# 不可當「行首」的字(多為詞尾/虛字,放句首很怪)
NO_START = set("麼們嗎呢吧啊喔噢啦嘛呀耶欸哦囉著地得過個子兒的了")
# 不可當「行尾」的字(多為副詞/連接/介詞,後面一定還有字)
# 「於之其」修「來自於|OB車上」類切法;「才再」是掛尾單字(就又也跟或 原本就在)
NO_END = set("很太也都就又更最還不把被跟越比讓沒要想會能可和與在從對給每該並或而且但因所雖於之其才再")
# 掛尾連接詞(卡尾以這些 2 字詞收尾=語意懸空,重罰;切在其「前」仍由 CONJ2 加分)
DANGLE_TAIL2 = {"然後", "所以", "因為", "但是", "可是", "而且", "或是", "還是",
                "而是", "就是", "不過", "其實", "譬如", "比如", "如果", "甚至", "後來"}
DANGLE_PENALTY = 4.0

# === 中文斷詞(jieba):讓斷行只發生在「詞與詞之間」,不切斷詞 ===
_JIEBA = None
# 節目固定專有名詞(避免被斷開);各集來賓/品牌再由 add_words() 動態加入
_BASE_WORDS = ["印花樂", "過嗨乳牛", "郝慧川", "岳啟儒", "沈奕妤", "惡魔老闆",
               "我愛上班", "育成中心", "孵化器", "客製化", "大稻埕", "台灣八哥"]


def _jieba():
    global _JIEBA
    if _JIEBA is None:
        try:
            import jieba
            jieba.setLogLevel(20)
            big = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dict.txt.big")
            if os.path.exists(big):
                jieba.set_dictionary(big)   # 繁體詞典
            for w in _BASE_WORDS:
                jieba.add_word(w)
            _JIEBA = jieba
        except Exception:
            _JIEBA = False                  # 沒裝 jieba → 退回純規則
    return _JIEBA


def add_words(words):
    """加入自訂詞(來賓姓名、品牌、術語),避免被斷開。"""
    j = _jieba()
    if not j:
        return
    for w in words:
        w = (w or "").strip()
        if len(w) >= 2:
            j.add_word(w)


def word_break_ok(text):
    """回傳「可斷點」的字元索引集合(= 各詞的邊界);沒 jieba 則回傳 None。"""
    j = _jieba()
    if not j:
        return None
    pts, n = {0}, 0
    for tok in j.cut(text):
        n += len(tok)
        pts.add(n)
    return pts


def char_width(c):
    if c.isspace():
        return 0.0
    return 0.5 if ord(c) < 128 else 1.0


def _text_width(s):
    return sum(char_width(c) for c in s)


def balanced_split(chars, max_w=16.0, min_w=5.0):
    """chars: [(ch, start, end), ...]。回傳片段 [(a, b), ...](半開區間 index)。"""
    n = len(chars)
    text = [c[0] for c in chars]
    cum = [0.0] * (n + 1)
    for i, c in enumerate(text):
        cum[i + 1] = cum[i] + char_width(c)
    total = cum[n]
    if total <= max_w or n <= 1:
        return [(0, n)]

    wb = word_break_ok("".join(text))   # 可斷點(詞邊界);None=沒 jieba

    def boundary(idx):
        """切在 idx-1 與 idx 之間的好壞分數(越高越適合斷在這)。"""
        before, at = text[idx - 1], text[idx]
        b = 0.0
        if before in PARTICLE_END:
            b += 3.0
        if "".join(text[idx:idx + 2]) in CONJ2:
            b += 2.2
        if at in NO_START:
            b -= 4.0
        if before in NO_END:
            b -= 3.5
        if "".join(text[max(0, idx - 2):idx]) in DANGLE_TAIL2:
            b -= DANGLE_PENALTY          # 卡尾掛「然後/所以/因為…」→ 重罰
        if before.isascii() and before.strip() and at.isascii() and at.strip():
            b -= 5.0
        return b

    # 先算要切幾段、每段的理想長度(平衡),再於「上限內」挑最接近理想又邊界好的點
    k = max(2, math.ceil(total / max_w))
    target = total / k
    pieces = []
    start = 0
    while cum[n] - cum[start] > max_w:
        hi = start
        while hi + 1 <= n and cum[hi + 1] - cum[start] <= max_w:
            hi += 1
        lo = start + 1
        while lo < hi and cum[lo] - cum[start] < min_w:
            lo += 1
        best, best_score = hi, -1e18
        for idx in range(lo, hi + 1):
            w = cum[idx] - cum[start]
            score = boundary(idx) - abs(w - target) * 0.5   # 接近理想長度 + 好邊界
            if wb is not None and idx not in wb:            # 切在詞中間 → 重罰
                score -= 8.0
            rem = total - cum[idx]
            if 0 < rem < min_w:                             # 別讓後面剩碎片
                score -= 6.0
            if score > best_score:
                best_score, best = score, idx
        pieces.append((start, best))
        start = best
    pieces.append((start, n))

    # 結尾若太短,與前一段合併後從中間再平均切一次(仍保證 ≤ max_w)
    if len(pieces) >= 2:
        a, b = pieces[-1]
        if cum[b] - cum[a] < min_w:
            s, e = pieces[-2][0], b
            if cum[e] - cum[s] <= max_w:
                pieces[-2:] = [(s, e)]
            else:
                mid = (cum[s] + cum[e]) / 2
                valid = [i for i in range(s + 1, e)
                         if cum[i] - cum[s] <= max_w and cum[e] - cum[i] <= max_w]
                bi = max(valid, key=lambda i: -abs(cum[i] - mid) + boundary(i) * 0.6
                         + (0 if (wb is None or i in wb) else -8.0))
                pieces[-2:] = [(s, bi), (bi, e)]
    return pieces


if __name__ == "__main__":
    # 自我測試:用等距時間模擬,檢查斷點
    samples = [
        "哈囉大家好歡迎來到收聽本周我愛上班我是郝慧川",
        "會長長短短可是基本上不太會超過肩膀",
        "因為其實我們覺得圖像它就是沒有文字",
        "所以我大學雖然唸的是美術系是純藝術",
        "不是但是因為我對織品設計是很有興趣的",
        "什麼我們智慧結晶怎麼可以被剽竊我覺得",
        "創辦人 Emma 沈奕妤",
        # 新測例(魁哥集根因):定語「的」/「於」/掛尾連接詞
        "因為現場真的太吵的時候你根本聽不見耳機裡的聲音",
        "這個訊號其實是來自於OB車上的導播設備傳過來的",
        "他們家的隔音跟通風設備真的做得很不錯然後我們就決定租下來了",
        "那一場的觀眾少說有五百人甚至可能快要一千人了",   # 舊評分會切成「…甚至|可能…」
        "他們平常會用無線的麥克風或是直接拉一條線到控台",  # 舊評分會切成「無線的|麥克風」
    ]

    def split_text(s):
        chars = [(c, i, i + 1) for i, c in enumerate(s)]
        return ["".join(s[a:b]) for a, b in balanced_split(chars, 16.0)]

    for s in samples:
        print(" | ".join(split_text(s)))

    # 斷言:評分修正後不再出現的壞切點
    bad = []
    for s in samples:
        pieces = split_text(s)
        joined = "|".join(pieces)
        if "太吵的|時候" in joined:
            bad.append(f"定語的被切開:{joined}")
        for p in pieces[:-1]:                      # 最後一片是卡尾=句尾,不算掛尾
            if p.endswith("於"):
                bad.append(f"「於」掛尾:{joined}")
            if p[-2:] in DANGLE_TAIL2:
                bad.append(f"連接詞掛尾:{joined}")
    if bad:
        raise SystemExit("❌ self-test 失敗:\n" + "\n".join(bad))
    print("✅ self-test 通過(無 定語的/於/連接詞 掛尾)")
