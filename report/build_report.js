// Builds report/NLP_Assignment1_Report.docx from outputs/results.json and outputs/figures.
// Usage: node report/build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, ImageRun, PageBreak, Footer, PageNumber, LevelFormat,
  TabStopType,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const R = JSON.parse(fs.readFileSync(path.join(ROOT, "outputs/results.json"), "utf8"));
const FIG = (f) => path.join(ROOT, "outputs/figures", f);
const SPACE_URL = "https://huggingface.co/spaces/nebadan/amharic-next-word";

// ---------- formatting helpers ----------
const fmtInt = (x) => Math.round(x).toLocaleString("en-US");
const fmt = (x, d = 1) => (x === null || x === undefined ? "∞" : Number(x).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }));
const pct = (x, d = 1) => fmt(100 * x, d) + "%";
const sci = (x) => (x === 0 ? "0" : Number(x).toExponential(2));

const BODY = "Calibri";
const ETH = "Ebrima"; // Ethiopic-capable font on Windows; LibreOffice falls back to Noto Sans Ethiopic
const ETH_RE = /([ሀ-᎟ⶀ-⷟꬀-꬯]+(?:[\s፡-፨]+[ሀ-᎟ⶀ-⷟꬀-꬯]+)*)/;

function runs(text, opts = {}) {
  return String(text).split(ETH_RE).filter((s) => s.length).map((s) =>
    new TextRun({ text: s, font: ETH_RE.test(s) ? ETH : (opts.font || BODY), ...opts, ...(ETH_RE.test(s) ? { font: ETH } : {}) }));
}
// inline **bold** support
function rich(text, opts = {}) {
  const out = [];
  String(text).split(/(\*\*[^*]+\*\*)/).forEach((part) => {
    if (part.startsWith("**") && part.endsWith("**")) out.push(...runs(part.slice(2, -2), { ...opts, bold: true }));
    else if (part) out.push(...runs(part, opts));
  });
  return out;
}
const P = (text, opts = {}) => (lastWasNumbered = false, new Paragraph({ children: rich(text), spacing: { after: 120, line: 276 }, ...opts }));
const HEADINGS = [];
const H1 = (t) => { HEADINGS.push([1, t]); return new Paragraph({ heading: HeadingLevel.HEADING_1, children: runs(t), pageBreakBefore: true }); };
const H2 = (t) => { HEADINGS.push([2, t]); return new Paragraph({ heading: HeadingLevel.HEADING_2, children: runs(t) }); };
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: runs(t) });
const bullet = (t, level = 0) => (lastWasNumbered = false, new Paragraph({ numbering: { reference: "bullets", level }, children: rich(t), spacing: { after: 60 } }));
let listInstance = 0, lastWasNumbered = false;
const numbered = (t) => {
  if (!lastWasNumbered) listInstance += 1;
  lastWasNumbered = true;
  return new Paragraph({ numbering: { reference: "numbers", level: 0, instance: listInstance }, children: rich(t), spacing: { after: 60 } });
};
const _P = (...a) => { lastWasNumbered = false; return a; };
const caption = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 240 }, children: runs(t, { italics: true, size: 18, color: "000000" }) });
const code = (t) => new Paragraph({ shading: { type: ShadingType.CLEAR, fill: "F3F3F3" }, spacing: { after: 0 }, children: runs(t, { font: "Consolas", size: 18 }) });

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function figure(file, widthIn, cap) {
  const { w, h } = pngSize(file);
  const W = widthIn * 96;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(file), transformation: { width: W, height: Math.round(W * h / w) } })] }),
    caption(cap),
  ];
}

const CONTENT_W = 9026; // A4 with 1" margins, in DXA
const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(headers, rows, widths, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const scale = CONTENT_W / total;
  const w = widths.map((x) => Math.floor(x * scale));
  w[w.length - 1] += CONTENT_W - w.reduce((a, b) => a + b, 0);
  const cell = (t, i, head, shade) => new TableCell({
    borders, width: { size: w[i], type: WidthType.DXA },
    shading: undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({ alignment: (i > 0 && opts.rightAlign !== false && !head) ? (opts.align?.[i] || AlignmentType.RIGHT) : (opts.align?.[i] || AlignmentType.LEFT),
      children: runs(t, { size: opts.size || 19, bold: head, color: "000000" }) })],
  });
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: w,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r, k) => new TableRow({ children: r.map((t, i) => cell(t, i, false, k % 2 === 1)) }))],
  });
}
const gap = () => new Paragraph({ spacing: { after: 120 }, children: [] });

// ---------- numbers used in the text ----------
const S = R.corpus_stats;
const NG = R.ngram_counts;
const PP = R.perplexity;
const EX = R.extrinsic;
const LC = R.learning_curve;
const SPL = R.split;
const PR = R.pruning || [];
const ppKN = (n) => PP[n - 1]["Kneser-Ney"];
const ex = (n, k) => EX[n - 1][k];

// ---------- document body ----------
const body = [];

// Title page
body.push(
  new Paragraph({ spacing: { before: 600, after: 300 }, alignment: AlignmentType.CENTER,
    children: [new ImageRun({ type: "png", data: fs.readFileSync(path.join(__dirname, "aau_logo.png")), transformation: { width: 150, height: Math.round(150 * 423 / 427) } })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: runs("Addis Ababa University", { size: 28, bold: true }) }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: runs("College of Technology & Built Environment", { size: 24 }) }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 700 }, children: runs("School of Information Technology and Engineering", { size: 24 }) }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: runs("Natural Language Processing — Assignment 1", { size: 30, color: "000000" }) }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: runs("Amharic N-gram Language Models", { size: 48, bold: true, color: "000000" }) }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1200 }, children: runs("Corpus analysis, intrinsic and extrinsic evaluation, and a next-word suggestion system on the GPAC corpus", { size: 24, italics: true }) }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: runs("Group members:", { size: 24, bold: true }) }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: runs("Neibyu Daniel", { size: 24 }) }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: runs("Yonas Tadesse", { size: 24 }) }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 600 }, children: runs("Selamawit Siferh", { size: 24 }) }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: runs("Instructor: Fantahun B. (PhD)", { size: 24 }) }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: runs("October 2026", { size: 24 }) }),
);

// Contents
body.push(
  new Paragraph({ pageBreakBefore: true, children: runs("Contents", { size: 32, bold: true, color: "000000" }), spacing: { after: 200 } }),
);
const TOC_AT = body.length;

// Summary
body.push(H1("Summary"));
body.push(P(`We built n-gram language models (n = 1–4) on the General Purpose Amharic Corpus (GPAC): ${fmt(S["Raw file size (MB)"], 0)} MB of text, which after cleaning gives **${fmtInt(S["Sentences"])} sentences and ${fmtInt(S["Tokens (after cleaning)"])} tokens** with **${fmtInt(S["Vocabulary size (types)"])} distinct words**. All four tasks were completed and run locally on the full corpus:`));
body.push(bullet(`**Task 1 (n-gram model):** corpus statistics, n-grams for n = 1–4, n-gram and conditional probabilities, stop-word analysis, word clouds, sentence probabilities and random sentence generation.`));
body.push(bullet(`**Task 2 (intrinsic evaluation):** on held-out sentences, interpolated Kneser-Ney perplexity falls from **${fmt(ppKN(1), 0)} (unigram) to ${fmt(ppKN(2), 0)}, ${fmt(ppKN(3), 0)} and ${fmt(ppKN(4), 0)}** for n = 2, 3, 4. Unsmoothed MLE gives zero probability to ${pct(PP[3]["MLE: % zero-prob tokens"] / 100)} of test tokens at n = 4, and add-one smoothing gets worse as n grows.`));
body.push(bullet(`**Task 3 (extrinsic evaluation):** in three downstream tasks the ranking is the same as for perplexity. Top-5 next-word accuracy rises from ${pct(ex(1, "NWP top5"))} (n = 1) to ${pct(ex(4, "NWP top5"))} (n = 4); a predictive keyboard saves ${pct(ex(3, "KSS (top-5)"))} of keystrokes with a trigram; and a trigram model tells real sentences from corrupted ones ${pct(ex(3, "discrim. replace"), 0)}–${pct(ex(3, "discrim. shuffle"), 0)} of the time, depending on the corruption.`));
body.push(bullet(`**Task 4 (application):** an interactive Amharic next-word suggestion app (top-5 suggestions with probabilities, click to insert, mid-word completion), deployed on Hugging Face Spaces: ${SPACE_URL}`));
body.push(P(`One methodological finding: **${fmt(S["Repeated sentences (%)"], 1)}% of GPAC sentences are exact repeats.** Without removing them before the train/test split, the same sentences appear in both sets and the evaluation scores are inflated. All evaluation results in this report use the deduplicated data.`));

// 1. Introduction
body.push(H1("1. Introduction"));
body.push(P("A statistical language model assigns a probability to a sequence of words. An n-gram model approximates the probability of each word using only the previous n − 1 words (the Markov assumption):"));
body.push(P("P(w₁ … wₘ) ≈ ∏ᵢ P(wᵢ | wᵢ₋ₙ₊₁ … wᵢ₋₁),    with the MLE estimate  P(wᵢ | h) = C(h wᵢ) / C(h)", { alignment: AlignmentType.CENTER }));
body.push(P("Amharic is a Semitic language written in the Ge'ez (Ethiopic) script. Its rich morphology — prepositions, conjunctions, possessives and object markers attach to the word — produces a very large vocabulary, which makes the data-sparsity problem of n-gram models particularly visible. This assignment studies that problem on a large real corpus and turns the model into a practical predictive-text tool."));
body.push(H2("1.1 Corpus"));
body.push(P("GPAC (General Purpose Amharic Corpus) was created by Gereme, Zhu, Ayall and Alemu (Information 12(1):20, 2021) as part of their work on Amharic fake-news detection. It was collected mainly from Amharic news websites, blogs and other web sources. The version provided for the assignment is a single 1.06 GB UTF-8 text file without line breaks."));
body.push(H2("1.2 Implementation"));
body.push(P("All code was written for this assignment in Python (numpy) and runs locally. Standard Python dictionaries of tuples do not fit 50 million distinct 4-grams in memory, so each n-gram is stored as a 64-bit hash of its word ids in sorted numpy arrays; look-ups are binary searches and all probabilities for a test set are computed in vectorised form. Building the four orders on 77 M tokens takes about two minutes. The full notebook (all tasks, ~19 minutes) needs about 22 GB of RAM."));
body.push(table(["Module", "Purpose"], [
  ["amharic_lm/text.py", "normalisation, sentence splitting, tokenisation, stop-word list"],
  ["amharic_lm/corpus.py", "streaming preprocessing, id encoding, train/test split"],
  ["amharic_lm/ngram.py", "hashed n-gram tables; MLE, add-k and interpolated Kneser-Ney; prediction; generation"],
  ["amharic_lm/evaluate.py", "next-word prediction, keystroke savings, corrupted-sentence discrimination"],
  ["notebooks/Assignment1_Amharic_Ngram.ipynb", "all tasks with outputs"],
  ["app/app.py", "Gradio next-word suggestion app (Hugging Face Space)"],
], [3, 6], { rightAlign: false }));

// 2. Task 1
body.push(H1("2. Task 1 — N-gram Language Model"));
body.push(H2("1.1 Preprocessing and corpus statistics"));
body.push(P("The preprocessing pipeline:"));
body.push(numbered("**Unicode NFC normalisation.**"));
body.push(numbered("**Homophone normalisation.** Modern Amharic writes the same sound with several letters: ሀ/ሐ/ኀ, ሰ/ሠ, አ/ዐ and ጸ/ፀ (plus ኣ/ዓ → አ and ሃ → ሀ). All seven vowel orders of each family are mapped to one form, so e.g. ሐገር and ሀገር become the same word."));
body.push(numbered("**Sentence segmentation** on ። (full stop), ፡፡ (two word-separators — the most common sentence end in this web corpus), ፧, ?, ! and ፨. The 1 GB single-line file is read in 4-million-character chunks cut at sentence ends."));
body.push(numbered("**Tokenisation.** Ethiopic punctuation (፡ ፣ ፤ ፥ ፦) and ASCII punctuation become spaces; dots and slashes inside abbreviations are removed (ዓ.ም → አም, ዶ/ር → ዶር); numbers, including 1,000,000 and dates, become one <NUM> token; Latin-script text and symbols are dropped."));
body.push(gap());
const statRows = [
  ["Raw file size", fmt(S["Raw file size (MB)"], 0) + " MB"],
  ["Raw characters / whitespace tokens", `${fmtInt(S["Raw characters"])} / ${fmtInt(S["Raw whitespace tokens"])}`],
  ["Sentences (unique)", `${fmtInt(S["Sentences"])} (${fmtInt(S["Unique sentences"])})`],
  ["Repeated sentences", fmt(S["Repeated sentences (%)"], 1) + "%"],
  ["Tokens after cleaning", fmtInt(S["Tokens (after cleaning)"])],
  ["Vocabulary size (types)", fmtInt(S["Vocabulary size (types)"])],
  ["Type/token ratio", fmt(S["Type/token ratio"], 4)],
  ["Hapax legomena (words seen once)", `${fmtInt(S["Hapax legomena (freq = 1)"])} (${fmt(S["Hapax % of vocabulary"], 1)}% of vocabulary)`],
  ["Mean / median sentence length", `${fmt(S["Mean sentence length (tokens)"], 1)} / ${fmt(S["Median sentence length"], 0)} tokens`],
  ["Mean word length (by token / by type)", `${fmt(S["Mean word length (chars, by token)"], 2)} / ${fmt(S["Mean word length (chars, by type)"], 2)} characters`],
  ["Stop-word share of tokens", fmt(S["Stop-word tokens (%)"], 1) + "%"],
  ["Zipf slope (log-log rank/frequency)", fmt(S["Zipf slope"], 2)],
  ["Heaps' law fit V = K·N^β", `K = ${fmt(S["Heaps K"], 1)}, β = ${fmt(S["Heaps beta"], 2)}`],
];
body.push(table(["Statistic", "Value"], statRows, [4, 5]));
body.push(caption("Table 1. GPAC corpus statistics after preprocessing."));
body.push(...figure(FIG("corpus_stats.png"), 6.3, "Figure 1. Zipf plot, sentence-length distribution and the most frequent words."));
body.push(P(`The frequency distribution follows Zipf's law (slope ${fmt(S["Zipf slope"], 2)}). The vocabulary keeps growing with corpus size — Heaps' exponent β = ${fmt(S["Heaps beta"], 2)} — and ${fmt(S["Hapax % of vocabulary"], 0)}% of all word types occur only once. Both are typical of a morphologically rich language: one lemma appears in many inflected and prefixed forms (e.g. ኢትዮጵያ, የኢትዮጵያ, በኢትዮጵያ, ለኢትዮጵያ, ኢትዮጵያን …). The most frequent tokens are function words such as ነው (is), ላይ (on), ውስጥ (in), ወደ (to) and እና (and).`));
body.push(P(`GPAC is crawled from the web, and **${fmt(S["Repeated sentences (%)"], 1)}% of its sentences are exact repeats** (agency datelines such as “… ኢሳት ዜና”, re-published articles). Task 1 describes the corpus as given; for evaluation (Tasks 2–4) each sentence is kept once.`));

body.push(H2("1.2 Creating n-grams (n = 1, 2, 3, 4)"));
body.push(P("Each sentence is padded with n − 1 start symbols <s> and one end symbol </s>; n-grams never cross sentence boundaries. For example, the bigrams of “ኢትዮጵያ በአፍሪካ ቀንድ ትገኛለች” are <s>_ኢትዮጵያ, ኢትዮጵያ_በአፍሪካ, በአፍሪካ_ቀንድ, ቀንድ_ትገኛለች, ትገኛለች_</s>. Sample n-grams of the first corpus sentence are printed in the notebook."));
body.push(table(["n", "Distinct n-grams", "Total n-grams", "Seen only once"],
  NG.map((r) => [String(r.n), fmtInt(r["unique n-grams"]), fmtInt(r["total n-grams"]), fmt(r["singletons (%)"], 1) + "%"]), [1, 3, 3, 3]));
body.push(caption("Table 2. Number of n-grams in the full corpus."));
body.push(P(`The number of distinct n-grams grows quickly with n: there are ${fmt(NG[3]["unique n-grams"] / 1e6, 1)} million distinct 4-grams in ${fmt(NG[3]["total n-grams"] / 1e6, 1)} million positions, and ${fmt(NG[3]["singletons (%)"], 0)}% of them are seen only once. Most 4-gram contexts therefore carry almost no statistical evidence, which is the root of the sparsity results in Tasks 2 and 3.`));

body.push(H2("1.3 N-gram probabilities and the top-10 n-grams"));
body.push(P("The probability of an n-gram is its relative frequency among all n-grams of the same order, P(w₁…wₙ) = C(w₁…wₙ) / Σ C(·). Sentence-boundary symbols are excluded from the lists."));
for (const n of [1, 2, 3, 4]) {
  const rows = R.top_ngrams[n].map((r, i) => [String(i + 1), r[0], fmtInt(r[1]), sci(r[2])]);
  body.push(table(["#", `${n}-gram`, "Count", "P"], rows, [0.6, 5, 1.6, 1.6], { align: [AlignmentType.LEFT, AlignmentType.LEFT] }));
  body.push(caption(`Table ${2 + n}. Top 10 ${n}-grams (full corpus, with stop-words).`));
}
body.push(P("Unigrams are dominated by function words and the number token. The higher-order lists are dominated by **dates and numbers** — ቀን <NUM> አ ም (“day … E.C.”), በ <NUM> አ ም, እ ኤ አ (“G.C.”) — because news text dates nearly every event. The fact that አ ም still appears as two tokens shows that the abbreviation ዓ.ም is often written with spaces (ዓ ም) in the source text."));

body.push(H2("1.4 Conditional probability of a word given the previous word"));
body.push(P("Using bigram counts, P(wᵢ | wᵢ₋₁) = C(wᵢ₋₁ wᵢ) / C(wᵢ₋₁ ·):"));
body.push(table(["Previous word", "Word", "C(prev word)", "C(prev ·)", "P(word | prev)"],
  R.conditional.map((r) => [r[0], r[1], fmtInt(r[2]), fmtInt(r[3]), fmt(r[4], 4)]), [2, 2, 1.6, 1.6, 1.8], { align: [AlignmentType.LEFT, AlignmentType.LEFT] }));
body.push(caption("Table 7. Bigram conditional probabilities (MLE)."));
body.push(P("Most likely next words for a few contexts:"));
body.push(table(["Previous word", "Most likely next words (P)"],
  Object.entries(R.followers).map(([w, f]) => [w, f.map(([x, p]) => `${x} (${fmt(p, 3)})`).join(",  ")]), [1.5, 7.5], { rightAlign: false }));
body.push(caption("Table 8. Top followers by bigram probability."));
body.push(P("Strong collocations such as ጠቅላይ → ሚኒስትር (prime → minister) and አዲስ → አበባ (Addis → Ababa) have high conditional probabilities. The pair ቀንድ → ትገኛለች (“horn → is located”) was **never observed**, so its MLE probability is exactly zero — even though it is perfectly grammatical. This is the main weakness of unsmoothed MLE estimates."));

body.push(H2("1.5 N-gram frequencies after stop-word removal"));
body.push(P(`We built a list of 163 Amharic function words — copulas (ነው, ናቸው, ነበር), conjunctions (እና, ግን, ወይም), postpositions (ላይ, ውስጥ, ጋር), demonstratives (ይህ, እነዚህ), pronouns, quantifiers and common reporting verbs (ገልጸዋል, ተናግረዋል) — plus <NUM>. Removing them deletes ${fmtInt(R.stopword_tokens_removed)} tokens (${fmt(S["Stop-word tokens (%)"], 1)}% of the corpus); n-grams are then rebuilt over the remaining words.`));
for (const n of [1, 2, 3, 4]) {
  const rows = R.top_ngrams_nostop[n].map((r, i) => [String(i + 1), r[0], fmtInt(r[1]), sci(r[2])]);
  body.push(table(["#", `${n}-gram`, "Count", "P"], rows, [0.6, 5, 1.6, 1.6], { align: [AlignmentType.LEFT, AlignmentType.LEFT] }));
  body.push(caption(`Table ${8 + n}. Top 10 ${n}-grams after stop-word removal.`));
}
body.push(P("After removal the lists show content words and multi-word names: places (አዲስ አበባ), institutions (ምክር ቤት, የህዝብ ተወካዮች ምክር ቤት — House of Peoples' Representatives; ፍርድ ቤት — court), titles with names (ጠቅላይ ሚኒስትር አብይ አህመድ, ጠቅላይ ሚኒስትር መለስ ዜናዊ), and news-agency boilerplate (… ኢሳት ዜና). The 3- and 4-gram lists now read like a topic index of Ethiopian news."));

body.push(H2("1.6 Word clouds before and after stop-word removal"));
body.push(...figure(FIG("wordclouds.png"), 5.6, "Figure 2. Word clouds of the 150 most frequent n-grams, n = 1–4 (left: with stop-words, right: stop-words removed)."));
body.push(P("With stop-words the clouds are dominated by grammatical words and date patterns; after removal, named entities and topics (government, people, Addis Ababa, the Prime Minister, parliament, church) become visible."));

body.push(H2("1.7 Probability of a sentence"));
body.push(P("The sentence probability is the product of the conditional probabilities of each word (and </s>) given the previous n − 1 words. We report log₁₀ probabilities for unsmoothed MLE and for interpolated Kneser-Ney (KN, Section 3), which never assigns zero, plus the per-word perplexity under KN."));
const spRows = R.sentence_probs.map((r) => [r[0], String(r[1]), r[2] === null ? "−∞ (P = 0)" : fmt(r[2], 2), fmt(r[4], 2), fmt(r[5], 0)]);
body.push(table(["Sentence", "n", "log₁₀ P (MLE)", "log₁₀ P (KN)", "PP (KN)"], spRows, [4.6, 0.5, 1.5, 1.4, 1.2], { size: 17, align: [AlignmentType.LEFT] }));
body.push(caption("Table 13. Sentence probabilities. The last sentence is the first one with its words reversed (control)."));
const B = R.sentence_breakdown;
body.push(table(["Trigram factor", "MLE", "Kneser-Ney"], B.map((r) => [r[0], fmt(r[1], 4), r[2] < 0.001 ? sci(r[2]) : fmt(r[2], 4)]), [5, 2, 2], { align: [AlignmentType.LEFT] }));
body.push(caption("Table 14. Trigram factors of P(ኢትዮጵያ በአፍሪካ ቀንድ ትገኛለች)."));
const s1 = R.sentence_probs.filter((r) => r[0].startsWith("ኢትዮጵያ በአፍሪካ"));
const s5 = R.sentence_probs.filter((r) => r[0].startsWith("ትገኛለች"));
body.push(P(`For “ኢትዮጵያ በአፍሪካ ቀንድ ትገኛለች” (Ethiopia is located in the Horn of Africa) the unigram probability is 10^${fmt(s1[0][2], 1)}. For n ≥ 2 the **MLE probability is zero**, because the bigram ቀንድ ትገኛለች never occurs in the corpus (Table 14). Kneser-Ney backs off to the probability of ትገኛለች as a continuation, giving P(sentence) ≈ 10^${fmt(s1[2][4], 1)} with a trigram. The reversed control sentence has the same unigram probability (a unigram model ignores word order), but under the bigram–4-gram models it is about 10^${fmt(s5[2][4] - s1[2][4], 0).replace("-", "")} times less likely, so the higher-order models clearly prefer the grammatical order. Longer common sentences such as “አዲስ አበባ የኢትዮጵያ ዋና ከተማ ናት” are seen in the corpus and get a non-zero MLE probability even at n = 4.`));

body.push(H2("1.8 Random sentence generation"));
body.push(P("Sentences were generated with Shannon's method: start from <s>, repeatedly sample the next word from the MLE distribution P(w | previous n − 1 words) and stop at </s> (or after 30 words)."));
for (const n of [1, 2, 3, 4]) {
  body.push(H3(`${n}-gram`));
  R.generated[n].slice(0, 3).forEach((s) => body.push(bullet(s + " ።")));
}
body.push(P("**What happens as n increases:**"));
body.push(bullet("**Unigram:** words are independent draws by frequency. The output is a bag of frequent words with no syntax and random length."));
body.push(bullet("**Bigram:** adjacent word pairs are grammatical (possessive + noun, noun + postposition), but the sentence drifts to a new topic every word or two and rarely forms a complete clause."));
body.push(bullet("**Trigram:** phrases of 3–5 words are fluent and many sentences end with a finite verb (ነበር, ይገኛል, ገልጿል) — the normal Amharic subject–object–verb ending. Coherence still breaks over longer spans."));
body.push(bullet("**4-gram:** output looks like real news sentences — because large parts **are** real sentences. Most 4-gram contexts occurred only once or twice (70% of 4-grams are singletons), so sampling copies long stretches of training text verbatim."));
body.push(P("Increasing n trades novelty for fluency: local grammaticality improves quickly, but the number of possible contexts grows exponentially, so beyond n = 3 the model mostly memorises its training data instead of generalising."));

// 3. Task 2
body.push(H1("3. Task 2 — Intrinsic Evaluation (Perplexity)"));
body.push(H2("3.1 Setup"));
body.push(P(`The corpus was deduplicated (each distinct sentence kept once), then split at random into 98% training and 2% test sentences (seed 42); 0.5% of training was held out as a development set to tune add-k. The vocabulary contains every word seen at least twice in training (${fmtInt(SPL["vocabulary (min count 2)"])} words); other words map to <UNK>, so the model has a probability for unknown words. The test OOV rate is ${fmt(SPL["test OOV rate (%)"], 2)}%.`));
body.push(table(["", "Sentences", "Tokens"], [
  ["Training", fmtInt(SPL["train sentences"]), fmtInt(SPL["train tokens"])],
  ["Development", fmtInt(SPL["dev sentences"]), "—"],
  ["Test", fmtInt(SPL["test sentences"]), fmtInt(SPL["test tokens"])],
], [3, 3, 3]));
body.push(caption("Table 15. Data split (after deduplication)."));
body.push(P("Perplexity is the inverse probability of the test set normalised by the number of predicted tokens N (all words plus </s>): PP = exp(−(1/N) Σ ln P(wᵢ | hᵢ)). Lower is better. Four estimators were compared for each order:"));
body.push(bullet("**MLE** — no smoothing."));
body.push(bullet("**Laplace (add-1):** (C(h w) + 1) / (C(h) + V)."));
body.push(bullet(`**Add-k:** the same with k tuned on the development set (best k = ${Object.values(R.best_k).map((k) => Number(k).toExponential(0)).join(", ")} for n = 1–4).`));
body.push(bullet("**Interpolated Kneser-Ney:** P(w|h) = max(C(h w) − D, 0)/C(h) + D·N₁₊(h •)/C(h) · P_KN(w | shorter h), with D = n₁/(n₁ + 2n₂) per order and continuation counts N₁₊(• h w) for the lower orders."));
body.push(H2("3.2 Results"));
body.push(table(["n", "MLE", "MLE zero-prob tokens", "Laplace (add-1)", "Add-k (tuned)", "Kneser-Ney"],
  PP.map((r) => [String(r.n), r.MLE === null ? "∞" : fmt(r.MLE, 0), fmt(r["MLE: % zero-prob tokens"], 1) + "%", fmt(r["Laplace (add-1)"], 0), fmt(r["add-k (tuned)"], 0), fmt(r["Kneser-Ney"], 1)]),
  [0.6, 1.3, 1.7, 1.6, 1.6, 1.5]));
body.push(caption("Table 16. Test-set perplexity."));
body.push(...figure(FIG("perplexity.png"), 6.3, "Figure 3. Perplexity by order and smoothing (left, log scale) and the share of test tokens that MLE gives zero probability (right)."));
body.push(H2("3.3 Effect of training-set size"));
body.push(table(["Training tokens", "n = 1", "n = 2", "n = 3", "n = 4"],
  LC.map((r) => [fmt(r["train tokens (M)"], 1) + " M", fmt(r["n=1"], 0), fmt(r["n=2"], 0), fmt(r["n=3"], 0), fmt(r["n=4"], 0)]), [2, 1.5, 1.5, 1.5, 1.5]));
body.push(caption("Table 17. Kneser-Ney test perplexity vs. training-set size."));
body.push(...figure(FIG("learning_curve.png"), 4.3, "Figure 4. Learning curves."));
body.push(H2("3.4 Discussion"));
body.push(bullet(`**MLE cannot be evaluated** beyond unigrams: ${fmt(PP[1]["MLE: % zero-prob tokens"], 0)}% of test tokens get zero probability with bigrams and ${fmt(PP[3]["MLE: % zero-prob tokens"], 0)}% with 4-grams, so perplexity is infinite. This is the sparsity problem measured directly.`));
body.push(bullet(`**Add-one smoothing gets worse as n grows** (${fmt(PP[0]["Laplace (add-1)"], 0)} → ${fmt(PP[3]["Laplace (add-1)"], 0)}). With V ≈ 1 million, adding 1 to every possible continuation gives almost all probability mass to unseen events. Tuning k down to 10⁻⁴–10⁻⁵ helps a lot, but add-k still peaks at the bigram and then degrades, because it never uses lower-order information.`));
body.push(bullet(`**Kneser-Ney improves at every order:** ${fmt(ppKN(1), 0)} → ${fmt(ppKN(2), 0)} → ${fmt(ppKN(3), 0)} → ${fmt(ppKN(4), 0)}. Interpolation lets an unseen 4-gram fall back on trigram, bigram and unigram evidence. The gain from 3 to 4 (${pct(1 - ppKN(4) / ppKN(3), 0)}) is smaller than from 2 to 3 (${pct(1 - ppKN(3) / ppKN(2), 0)}): diminishing returns.`));
body.push(bullet(`**More data helps higher orders most.** With ${fmt(LC[0]["train tokens (M)"], 1)} M training tokens the 4-gram is no better than the trigram (${fmt(LC[0]["n=4"], 0)} vs ${fmt(LC[0]["n=3"], 0)}); with ${fmt(LC[3]["train tokens (M)"], 0)} M tokens it is ${pct(1 - LC[3]["n=4"] / LC[3]["n=3"], 0)} better. The unigram hardly changes. The amount of data, not the model, limits higher-order n-grams.`));
body.push(bullet(`The perplexities are much higher than typical values for English n-gram models, whose vocabularies are far smaller. With ${fmtInt(SPL["vocabulary (min count 2)"])} word types, every Amharic prediction has many more alternatives; sub-word units (morphemes or BPE) would lower this considerably.`));
body.push(bullet(`**Deduplication matters.** In a first run on the raw (non-deduplicated) data the 4-gram KN perplexity was about 111 instead of ${fmt(ppKN(4), 0)}, because a quarter of the sentences are repeats and many test sentences had been seen in training. The honest numbers are the ones above.`));

// 4. Task 3
body.push(H1("4. Task 3 — Extrinsic Evaluation"));
body.push(P("Perplexity measures how well a model predicts text, but not whether it helps in an application. We placed each Kneser-Ney model (n = 1–4) into three downstream tasks, all on the held-out test sentences:"));
body.push(numbered("**Next-word prediction** — the task behind Task 4. For 5,000 random test positions the model ranks the whole vocabulary given the history; we report top-1/3/5/10 accuracy and mean reciprocal rank (MRR@10)."));
body.push(numbered("**Keystroke savings (KSS)** in a simulated predictive keyboard. For 1,500 test words the user types one letter at a time until the intended word appears among 5 suggestions, then taps it (1 keystroke; the space is automatic). KSS = 1 − keystrokes used / keystrokes needed without prediction."));
body.push(numbered("**Real vs. corrupted sentence discrimination** — a fluency / grammaticality judgement of the kind used in spelling and grammar checking and for ranking speech-recognition or machine-translation outputs. For 3,000 test sentences the model must give the real sentence a higher probability than a copy with (a) two adjacent words swapped, (b) one word replaced by a random frequent word, or (c) all words shuffled. Chance level is 50%."));
body.push(table(["n", "Top-1", "Top-3", "Top-5", "Top-10", "MRR@10", "KSS", "Swap", "Replace", "Shuffle"],
  EX.map((r) => [String(r.n), pct(r["NWP top1"]), pct(r["NWP top3"]), pct(r["NWP top5"]), pct(r["NWP top10"]), fmt(r["NWP mrr@10"], 3), pct(r["KSS (top-5)"]), pct(r["discrim. swap"]), pct(r["discrim. replace"]), pct(r["discrim. shuffle"])]),
  [0.5, 1, 1, 1, 1, 1.1, 1, 1, 1.1, 1.1], { size: 17 }));
body.push(caption("Table 18. Extrinsic results (Kneser-Ney). Columns 2–6: next-word prediction; KSS: keystroke savings with 5 suggestions; last three: discrimination accuracy."));
body.push(...figure(FIG("extrinsic.png"), 6.3, "Figure 5. Extrinsic results by model order."));
body.push(...figure(FIG("intrinsic_vs_extrinsic.png"), 3.6, "Figure 6. Lower perplexity goes with higher next-word accuracy."));
body.push(H2("4.1 Discussion"));
body.push(bullet(`**Context is what makes prediction work.** Top-5 accuracy goes ${pct(ex(1, "NWP top5"))} → ${pct(ex(2, "NWP top5"))} → ${pct(ex(3, "NWP top5"))} → ${pct(ex(4, "NWP top5"))} for n = 1–4. The ranking of models is the same as with perplexity (Figure 6), so here the intrinsic measure was a good predictor of extrinsic usefulness.`));
body.push(bullet(`**Keystroke savings are large even for weak models** (${pct(ex(1, "KSS (top-5)"))} for the unigram, ${pct(ex(4, "KSS (top-5)"))} for the 4-gram). Amharic words are long (≈${fmt(S["Mean word length (chars, by type)"], 0)} characters per word type), and after one or two letters the prefix already narrows the candidates sharply.`));
body.push(bullet(`**The unigram model is at exactly 50% on swap and shuffle**: it ignores word order, so every permutation has the same probability. On the replace task it scores below chance (${pct(ex(1, "discrim. replace"))}) because the inserted words are frequent ones, which a unigram model prefers.`));
body.push(bullet(`**Discrimination difficulty depends on the corruption.** With a trigram, fully shuffled sentences are detected ${pct(ex(3, "discrim. shuffle"))} of the time, a single adjacent swap ${pct(ex(3, "discrim. swap"))} and a single replaced word ${pct(ex(3, "discrim. replace"))}. A replacement changes only a few n-grams, and a frequent word often fits its neighbours locally. This task saturates already at n = 2–3: judging word order is easier than predicting the exact next word.`));
body.push(bullet("Gains from 3 to 4 are smaller than from 2 to 3 in all tasks, consistent with the perplexity results and the sparsity of 4-grams."));

// 5. Task 4
body.push(H1("5. Task 4 — Amharic Predictive Text and Next-Word Suggestion"));
body.push(H2("5.1 Design"));
body.push(P("The suggestion engine follows the steps required in the assignment:"));
body.push(numbered("**Read the input** and normalise/tokenise it exactly like the training corpus. Only the sentence currently being written is used (text after the last ። or ፡፡)."));
body.push(numbered("**Extract the final one or two words** — trigram context by default; the user can switch to one word (bigram) or three words (4-gram). At the start of a sentence the context is padded with <s>."));
body.push(numbered("**Retrieve candidate next words** — every word observed after the context at each order (via a context index over the n-gram tables), plus the 300 most frequent words as a fallback."));
body.push(numbered("**Estimate the probability** of each candidate with interpolated Kneser-Ney, so unseen contexts back off smoothly instead of producing no suggestions."));
body.push(numbered("**Return the top five** suggestions with their probabilities. If the last word is only partly typed (no trailing space), only words starting with those letters are considered — word completion."));
body.push(numbered("**Let the user select a suggestion and continue** — clicking a suggestion inserts it (replacing the partial word) and the suggestions update immediately. ። is offered when the model predicts the end of the sentence; a button can also complete the sentence greedily."));
body.push(H2("5.2 Example suggestions"));
body.push(table(["Input", "Top-5 suggestions (P)"],
  Object.entries(R.suggestions).map(([q, s]) => [q.trim() + " …", s.map(([w, p]) => `${w} (${fmt(p, 3)})`).join(",  ")]), [2.4, 6.6], { rightAlign: false }));
body.push(caption("Table 19. Suggestions from the full trigram model."));
const coffee = R.suggestions["ኢትዮጵያ ውስጥ የቡና "].map(([w]) => w);
body.push(P(`For the assignment's example “ኢትዮጵያ ውስጥ የቡና …” the model suggests ${coffee.join(", ")} — all plausible continuations about coffee (consumption, price, export, production, trade). ምርት (production), the assignment's own example, is ranked ${coffee.indexOf("ምርት") + 1}${["st","nd","rd","th","th"][coffee.indexOf("ምርት")]}: it is the most frequent continuation of የቡና as a bigram (Table 8), but with the extra context ውስጥ the trigram prefers other words, so the longer context changes the ranking.`));
body.push(H2("5.3 Deployment"));
body.push(P(`There are two versions of the app. The **local version** (app/app.py) is a Gradio app that loads the 4-gram model and computes exact Kneser-Ney probabilities for every query. Hugging Face now requires a paid plan to host Gradio apps, so the **hosted version** is a free static Space that runs entirely in the browser: ${SPACE_URL}.`));
body.push(P(`To keep the files small, the model is first pruned: bigrams, trigrams and 4-grams seen fewer than 3 times are removed (all unigrams are kept), giving a ${fmt(R.app_model_mb, 0)} MB model instead of ~1 GB. From this model we precompute the top-8 Kneser-Ney suggestions for the 178,429 one-word contexts seen at least 10 times and the 222,085 two-word contexts seen at least 20 times, plus a 239,185-word vocabulary (words seen at least 10 times) for word completion — a 17 MB compressed file. The JavaScript page applies the same normalisation and tokenisation as the Python code, looks up the two-word context, backs off to the one-word context and then to word frequency (backed-off scores are marked “~”), and filters by the typed prefix.`));
if (PR.length) {
  body.push(table(["Model", "n", "Top-1", "Top-5", "MRR@10", "ms / query"],
    PR.map((r) => [r.model, String(r.n), pct(r.top1), pct(r.top5), fmt(r["mrr@10"], 3), fmt(r.ms_per_query, 1)]), [2.2, 0.6, 1.2, 1.2, 1.2, 1.4], { align: [AlignmentType.LEFT] }));
  body.push(caption("Table 20. Next-word prediction with the full and the pruned (deployed) model."));
}
const shot = FIG("app_screenshot.png");
if (fs.existsSync(shot)) body.push(...figure(shot, 5.2, "Figure 7. The hosted next-word suggestion app for the input “ኢትዮጵያ ውስጥ የቡና”."));
body.push(P("To run the exact version locally: python app/app.py (it loads app/model/amharic_4gram_kn.npz written by the notebook). The static tables are rebuilt with python static_app/export_tables.py and published with static_app/deploy.sh."));

// 6. Conclusion
body.push(H1("6. Limitations and Future Work"));
body.push(bullet("**Word-level sparsity.** Amharic's ~1.8 M word types make n-gram counts sparse. Morpheme- or sub-word-level models (e.g. BPE, or a morphological analyser such as HornMorpho) would share statistics across inflected forms."));
body.push(bullet("**Normalisation choices.** Homophone normalisation merges spelling variants but also changes the displayed spelling of suggestions (e.g. ሐ → ሀ). Abbreviations written with spaces (ዓ ም) remain split."));
body.push(bullet("**Domain.** GPAC is mostly news and political commentary; suggestions reflect that domain (dates, officials, institutions)."));
body.push(bullet("**Pruning trade-off.** The deployed model is smaller and faster but less accurate than the full model (Table 20), and the static tables only cover frequent contexts. A paid or self-hosted server could run the full model with exact probabilities."));
body.push(bullet("**Neural models.** An LSTM or small Transformer language model would capture longer context; the n-gram models here are a strong, fast and interpretable baseline."));
body.push(H2("Conclusion"));
body.push(P(`On 77 M tokens of Amharic, n-gram models behave as theory predicts: higher orders capture more context but suffer from sparsity, unsmoothed and add-one estimates fail, and Kneser-Ney smoothing turns the extra context into steady gains (perplexity ${fmt(ppKN(1), 0)} → ${fmt(ppKN(4), 0)}). The extrinsic tasks confirm the same ranking. Used as a predictive keyboard, a trigram model ranks the intended word in its top five suggestions ${pct(ex(3, "NWP top5"), 0)} of the time and saves ${pct(ex(3, "KSS (top-5)"), 0)} of keystrokes.`));

body.push(H1("References"));
[
  "Gereme, F., Zhu, W., Ayall, T., & Alemu, D. (2021). Combating Fake News in “Low-Resource” Languages: Amharic Fake News Detection Accompanied by Resource Crafting. Information, 12(1), 20. https://doi.org/10.3390/info12010020",
  "Jurafsky, D., & Martin, J. H. (2024). Speech and Language Processing (3rd ed. draft), Chapter 3: N-gram Language Models.",
  "Kneser, R., & Ney, H. (1995). Improved backing-off for M-gram language modeling. Proc. ICASSP, 181–184.",
  "Chen, S. F., & Goodman, J. (1999). An empirical study of smoothing techniques for language modeling. Computer Speech & Language, 13(4), 359–394.",
  "Shannon, C. E. (1948). A Mathematical Theory of Communication. Bell System Technical Journal, 27, 379–423.",
].forEach((r) => body.push(P(r)));

// ---------- table of contents (page numbers from a first render) ----------
const pagesFile = path.join(__dirname, ".toc_pages.json");
const pages = fs.existsSync(pagesFile) ? JSON.parse(fs.readFileSync(pagesFile, "utf8")) : {};
const tocParas = HEADINGS.map(([lvl, t]) => new Paragraph({
  indent: { left: lvl === 2 ? 360 : 0 }, spacing: { after: lvl === 1 ? 80 : 40, before: lvl === 1 ? 80 : 0 },
  tabStops: [{ type: TabStopType.RIGHT, position: CONTENT_W, leader: "dot" }],
  children: [...runs(t, { bold: lvl === 1, size: lvl === 1 ? 22 : 21 }),
    new TextRun({ text: "\t" + String(pages[t] || ""), size: lvl === 1 ? 22 : 21, bold: lvl === 1 })],
}));
body.splice(TOC_AT, 0, ...tocParas);
fs.writeFileSync(path.join(__dirname, ".headings.json"), JSON.stringify(HEADINGS));

// ---------- document ----------
const doc = new Document({
  creator: "NLP Assignment 1 group",
  title: "Amharic N-gram Language Models",
  styles: {
    default: { document: { run: { font: BODY, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 34, bold: true, font: BODY, color: "000000" }, paragraph: { spacing: { before: 240, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 27, bold: true, font: BODY, color: "000000" }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, font: BODY, color: "000000" }, paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } },
        { level: 1, format: LevelFormat.BULLET, text: "◦", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 300 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 340 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "000000" })] })] }) },
    children: body,
  }],
});

const out = path.join(__dirname, "NLP_Assignment1_Report.docx");
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("wrote", out); });
