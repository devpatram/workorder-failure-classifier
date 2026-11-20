# Results

Test set: 120 made-up work orders, 60 with wording seen in training and 60 with new wording. Trained on 240. LLM: Qwen/Qwen2.5-1.5B-Instruct.

| Method | Wording | Both labels right | Component | Failure | Labeled automatically | Right when labeled |
| --- | --- | --- | --- | --- | --- | --- |
| Keyword rules | seen | 100% | 100% | 100% | 100% | 100% |
| Keyword rules | new | 0% | 0% | 0% | 0% | n/a |
| Keyword rules | all | 50% | 50% | 50% | 50% | 100% |
| TF-IDF + logistic regression | seen | 100% | 100% | 100% | 100% | 100% |
| TF-IDF + logistic regression | new | 7% | 32% | 25% | 100% | 7% |
| TF-IDF + logistic regression | all | 53% | 66% | 62% | 100% | 53% |
| TF-IDF, review below 0.6 confidence | seen | 82% | 92% | 88% | 82% | 100% |
| TF-IDF, review below 0.6 confidence | new | 0% | 5% | 3% | 0% | n/a |
| TF-IDF, review below 0.6 confidence | all | 41% | 48% | 46% | 41% | 100% |
| LLM (Qwen/Qwen2.5-1.5B-Instruct) + validation | seen | 72% | 78% | 75% | 85% | 84% |
| LLM (Qwen/Qwen2.5-1.5B-Instruct) + validation | new | 43% | 57% | 58% | 82% | 53% |
| LLM (Qwen/Qwen2.5-1.5B-Instruct) + validation | all | 57% | 68% | 67% | 83% | 69% |
| Keyword rules, then the LLM for the rest | seen | 100% | 100% | 100% | 100% | 100% |
| Keyword rules, then the LLM for the rest | new | 43% | 57% | 58% | 82% | 53% |
| Keyword rules, then the LLM for the rest | all | 72% | 78% | 79% | 91% | 79% |

## The LLM's answers that went to review

20 of 120 failed validation and went to a person. 31 passed validation but were wrong.

| Work order | Text | Why it went to review |
| --- | --- | --- |
| WO20000 | Ops reported gb-12 down, bearing worn, parts ordered | unknown failure 'worn' |
| WO20057 | ops reported M-4 down, belt misaligned, adjsuted | unknown failure 'misaligned' |
| WO20045 | v-belt on GB-12 came loose, repl | unknown component 'v-belt' |
| WO20047 | Ops reported p-101 down, bearing worn out, temp fix, needs pm | unknown failure 'worn out' |
| WO20050 | found drive belt cracked on CV-5, temp fix, needs PM | unknown failure 'cracked' |
| WO20013 | F-2: mech seal leaking oil - adjusted | unknown component 'mech seal' |
| WO20037 | CV-5: v-belt showing wear - parts ordered | unknown component 'v-belt' |
| WO20001 | OPS REPORTED P-101 DOWN, DRIVE END BEARING RUNNING HOT, PARTS ORDEERD | unknown component 'drive end bearing' |
| WO20030 | M-4: v-belt snapped - claened | unknown component 'v-belt' |
| WO30054 | F-2 timing strap pitted. repl | unknown component 'timing strap'; unknown failure 'pitted' |
| WO30004 | Timing strap on ex-1 chewed up, wo to follow | unknown component 'timing strap'; unknown failure 'chewed up' |
| WO30032 | Cv-5: gear case weeping - tagged out | evidence is not quoted from the work order |
| WO30055 | CV-5 timing strap split. tightened | unknown component 'timing strap' |
| WO30048 | CV-3 packing gland pitted. tihgtened | unknown component 'packing gland'; unknown failure 'pitting' |
| WO30025 | Ops reported pr-9 down, spider insert wobbling, wo to follow | unknown component 'spider insert'; unknown failure 'wobbling' |
