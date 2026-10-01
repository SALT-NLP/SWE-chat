# SWE-chat: Coding Agent Interactions From Real Users in the Wild

[![COLM 2026](https://img.shields.io/badge/COLM-2026-6b4fbb.svg)](https://arxiv.org/abs/2604.20779)
[![arXiv](https://img.shields.io/badge/arXiv-2604.20779-b31b1b.svg)](https://arxiv.org/abs/2604.20779)
[![Website](https://img.shields.io/badge/Website-swe--chat.com-blue)](https://www.swe-chat.com/)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/SALT-NLP/SWE-chat)

**Paper:** [SWE-chat: Coding Agent Interactions From Real Users in the Wild](https://arxiv.org/abs/2604.20779) (COLM 2026)

**Dataset:** [SALT-NLP/SWE-chat on HuggingFace](https://huggingface.co/datasets/SALT-NLP/SWE-chat)

**Website:** [swe-chat.com](https://www.swe-chat.com/)

---

SWE-chat is a continually growing dataset of real coding agent sessions collected from open-source developers in the wild. The v2 snapshot (data through September, 2026) contains **17,819 sessions** across
**707 public GitHub repositories**, comprising **229,909 user prompts** and **2,028,951 agent tool calls**, with human vs. agent code authorship attribution.

## Demo of data analyses

[`analyses_and_figures.ipynb`](analyses_and_figures.ipynb) is a short tour of the dataset. It contains statistics and figures, starting from the released dataset, which it downloads from HuggingFace. First, request access on the [dataset page](https://huggingface.co/datasets/SALT-NLP/SWE-chat) and log in with a Hugging Face token from an account with access.

```bash
pip install -r requirements.txt
hf auth login
jupyter lab analyses_and_figures.ipynb
```

The notebook needs about 4 GB of working memory; allow additional headroom and roughly 7 GB of disk space for the downloaded tables. For local data, set `SWECHAT_LOCAL_DIR` to the directory containing the parquet files.

## Citation

```bibtex
@inproceedings{baumann2026swechat,
  title     = {SWE-chat: Coding Agent Interactions From Real Users in the Wild},
  author    = {Baumann, Joachim and Padmakumar, Vishakh and Li, Xiang and
               Yang, John and Yang, Diyi and Koyejo, Sanmi},
  booktitle = {Third Conference on Language Modeling},
  year      = {2026},
  url       = {https://arxiv.org/abs/2604.20779}
}
```
