---
title: Amharic Next-Word Suggestion
emoji: ✍️
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: mit
short_description: Amharic predictive text with a Kneser-Ney 4-gram model
---

# የአማርኛ ቃል ጠቋሚ · Amharic Next-Word Suggestion

Predictive text for Amharic built on an n-gram language model trained on the
**GPAC** corpus (General Purpose Amharic Corpus, Gereme et al. 2021, ~77M tokens).

**How it works**
1. Reads and normalizes your input exactly like the training corpus.
2. Takes the last one or two words (bigram / trigram context; 4-gram selectable).
3. Retrieves every word that followed that context in the corpus, plus frequent fallbacks.
4. Scores each candidate with interpolated Kneser-Ney smoothing.
5. Shows the top 5 — click one to add it and continue. Stopping mid-word completes the word.

NLP Assignment 1 — Addis Ababa University, School of Information Technology and Engineering.
