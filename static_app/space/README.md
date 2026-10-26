---
title: Amharic Next-Word Suggestion
emoji: ✍️
colorFrom: green
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Amharic predictive text with a Kneser-Ney n-gram model
---

# የአማርኛ ቃል ጠቋሚ · Amharic Next-Word Suggestion

Predictive text for Amharic. Type an incomplete sentence; the app takes the last one or two
words, looks up candidate next words with their interpolated **Kneser-Ney** probabilities, and
shows the top five. Click one to insert it and keep writing; stop mid-word to complete the word.

The suggestion tables were precomputed from a 4-gram model trained on the deduplicated **GPAC**
corpus (General Purpose Amharic Corpus, Gereme et al. 2021): 239k-word vocabulary,
178k one-word and 222k two-word contexts. Everything runs in the browser.

NLP Assignment 1 — Addis Ababa University, School of Information Technology and Engineering.
