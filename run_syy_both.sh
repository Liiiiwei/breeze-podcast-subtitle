#!/bin/bash
# 沈奕妤：兩種方式全長重跑 + 出 SRT + 對 fumo 比較。順序跑(避免兩個大模型同時吃爆記憶體)。
set -e
cd "/Users/Mac365/Developer/breeze subtitle/Breeze-ASR-25"
PY=./.venv/bin/python
DIR="/Users/Mac365/Downloads/20260508 沈奕妤"
FUMO="/Users/Mac365/Downloads/沈奕妤 Fumo srt.srt"
OUT=/tmp/syy_full
mkdir -p $OUT

echo "########## [1/2] STEREO 全長 ##########"
$PY asr_dump.py --mode stereo --dir "$DIR" --out $OUT/stereo.json 2>&1 | grep -vE 'frames|^\s*$' || true
$PY build_from_cache.py $OUT/stereo.json $OUT/stereo.srt $OUT/stereo.labeled.srt --corrections corrections_syy.json

echo "########## [2/2] PER-MIC 全長 ##########"
$PY asr_dump.py --mode permic --dir "$DIR" --out $OUT/permic.json 2>&1 | grep -vE 'frames|^\s*$' || true
$PY build_from_cache.py $OUT/permic.json $OUT/permic.srt $OUT/permic.labeled.srt --corrections corrections_syy.json

echo "########## 比較 vs fumo ##########"
echo "===== STEREO ====="
$PY /tmp/syy_compare.py $OUT/stereo.srt "$FUMO" "方式2 STEREO 全長"
echo "===== PER-MIC ====="
$PY /tmp/syy_compare.py $OUT/permic.srt "$FUMO" "方式1 PER-MIC 全長"
echo "ALL_DONE"
