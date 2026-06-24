"""遲滯貼標 hysteresis_spk 的合成驗證（不需音檔/模型）。

跑法：python test_hysteresis_spk.py
驗：a 講話中、b 出現的短暫小尖峰（<4dB）不該把講者翻成 b（串音/搶話峰值）；
   b 真的明顯大聲（>4dB）才換人。對照 margin=0（等同舊版瞬時 argmax）會被尖峰騙到。
"""
import numpy as np

from make_subtitle import hysteresis_spk

# frame_power 是每 0.05s 一框的線性功率；window_db = 10*log10(mean)
a = np.full(200, 1e-3)    # a 全程 -30dB
b = np.full(200, 1e-6)    # b 底噪 -60dB
b[40:46] = 10 ** (-2.8)   # 2.0–2.3s：b 短尖峰 -28dB（只比 a 大 2dB → 該被擋）
a[100:200] = 1e-6         # 5s 後 a 收聲
b[100:200] = 10 ** (-2.5)  # 5s 後 b 真的講 -25dB（比 a 大 35dB → 該換）

words = [
    {"start": 0.0, "end": 1.0}, {"start": 1.0, "end": 2.0},
    {"start": 2.0, "end": 2.3}, {"start": 2.3, "end": 3.0},  # 第 3 句 = b 尖峰
    {"start": 5.0, "end": 6.0}, {"start": 6.0, "end": 7.0},  # b 真的講
]
labels = ["a", "b"]
rp = [a, b]

got = [w["spk"] for w in hysteresis_spk([dict(w) for w in words], rp, labels, margin=4.0)]
assert got == ["a", "a", "a", "a", "b", "b"], f"遲滯應擋住尖峰、b 真大聲才換：{got}"

# 對照組：margin=0（等同瞬時 argmax）→ 尖峰那句會被翻成 b
got0 = [w["spk"] for w in hysteresis_spk([dict(w) for w in words], rp, labels, margin=0.0)]
assert got0[2] == "b", f"無遲滯時尖峰應翻 b（反證遲滯有效）：{got0}"

# 沒有原始麥時不動 spk（單軌相容）
none_in = [{"start": 0.0, "end": 1.0, "spk": None}]
assert hysteresis_spk(none_in, [], labels)[0]["spk"] is None

print("✅ hysteresis_spk OK：尖峰被遲滯擋住(a)、b 真大聲(>4dB)才換；對照 margin=0 會被尖峰翻成 b")
