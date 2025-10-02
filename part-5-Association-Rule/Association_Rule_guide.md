# Data Mining — Association Rule Learning

## What data mining is

<p align="center">
    <img src="img/DtxM7NLW4AI9uBZ.jpg" width="40%">
</p>

Data mining is the broader process of analyzing large datasets to extract useful patterns and relationships. One of its key branches is **association rule learning**, which focuses on finding interesting rules between items in transactional data. Instead of predicting labels or grouping points, it uncovers co‑occurrence patterns, like “customers who buy bread also often buy butter.”

## What association rule learning is
This family of models is about finding patterns like “people who buy X also tend to buy Y.” We search through transactions (shopping baskets, clicks, or any sets of items) to discover rules that describe associations. The goal isn’t prediction, but insight.

---

## Apriori

### The idea
Apriori generates itemsets step by step: first 1‑item sets, then 2‑item sets, then 3‑item sets, and so on. At each step it only keeps the frequent ones. The key property is: if an itemset is frequent, then all its subsets must also be frequent. This lets us prune the search space dramatically.

<p align="center">
    <img src="img/1b1X3sV7WgElbWUZCYMOMrA (1).webp" width="40%">
</p>

### The measures: support, confidence, lift
- **Support**: how often an itemset appears in the dataset. High support means it’s common.
- **Confidence**: how often the rule holds true. For rule X → Y, it’s the probability of Y given X.
- **Lift**: how much better the rule is than random chance. Lift > 1 means X and Y appear together more often than if they were independent.

We calculate these to decide which rules are worth keeping. Support ensures we only keep patterns that aren’t too rare. Confidence ensures the rule is reliable. Lift ensures the rule is truly meaningful, not just a coincidence of popularity.

### Steps
1. Set a minimum support and confidence.
2. Take all subsets in transactions that have support above the threshold.
3. From those, generate rules with confidence above the threshold.
4. Sort the rules by decreasing lift.

### Code
```python
import pandas as pd

dataset = pd.read_csv('data.csv', header=None)
transactions = []
for i in range(0, dataset.shape[0]):
    transactions.append([str(dataset.values[i, j]) for j in range(0, dataset.shape[1])])
```
Here we build a list of transactions, each transaction being a list of items. The loop goes through every row, collects its items, and stores them as a transaction. That’s why we use `append` with a list comprehension.

Once transactions are built, you can feed them into an Apriori implementation to generate rules.

---

## Eclat

### The idea
Eclat also finds frequent itemsets, but it does so differently. Instead of generating candidates level by level, it works by intersecting **transaction ID lists**. For each item, we store the list of transaction IDs where it appears. To check a combination of items, we intersect their lists. The support is simply the length of that intersection. This makes Eclat usually faster and more memory‑efficient than Apriori.

### Steps
1. Set a minimum support.
2. Take all subsets in transactions that have support above the threshold.
3. Sort the rules by decreasing support.

---

## Recap
- **Apriori**: bottom‑up, generates candidates, uses support → confidence → lift. Good for small to medium datasets, easy to interpret.
- **Eclat**: set‑intersection approach, usually faster. Focuses more on frequent itemsets and support, not on generating rules with lift.

