"""The KL v2 corpus builder moved to ``qda-kl corpus`` (``qda_kl.corpus``); this path keeps its old command line.

python tools/build_kl_corpus.py --out klv2-corpus.jsonl [--guard DIR]
"""

from qda_kl.corpus import main

if __name__ == "__main__":
    raise SystemExit(main())
