---
title: "From Zero to ML Engineer, Part 2: Tools and the First Data"
date: 2027-04-30 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, beginner, numpy, pandas, data, python, ML engineer]
description: "Part 2 of a 12-part series for complete beginners. The minimum of numpy and pandas needed to read a dataset, check it for problems, and see which columns carry information, with a 40-line program and three plots."
---

## What you will do, and what you get

You will load a dataset from a file, then answer in a few minutes the questions that decide how the rest of a project goes: how many rows, are any values missing, are the classes balanced, what range does each column cover, and which columns separate the classes. The data is the same 569-tumour table as Part 1. You need two libraries, installed once with `pip install numpy pandas scikit-learn matplotlib`.

![Three panels. Left: histograms of mean radius for benign and malignant rows, overlapping between 12 and 18, with a threshold line at 15. Middle: the largest value of each of the 30 columns on a log scale, from 0.03 to 4254. Right: the accuracy of the best single threshold on each column, from 92.3% for the best down to 62.7% for the worst, where always guessing benign gives 62.7%]({{ site.baseurl }}/assets/pic/mlpath-2-first-data.png){: width="900" }

## How: the steps

1. **Put the data in a table.** `pd.read_csv("file.csv")` gives a *DataFrame*: rows are examples, columns have names.
2. **Check the shape and the types.** `df.shape` is (rows, columns). Surprises here, such as far fewer rows than expected, are common and cheap to catch.
3. **Look for missing values and duplicates.** `df.isna().sum()` and `df.duplicated().sum()`.
4. **Count the classes.** `df["label"].value_counts()`. If 99% of rows are one class, accuracy will mislead you (Part 6).
5. **Read the ranges.** `df.describe()` shows the minimum, maximum and spread of each column.
6. **Compare the classes.** `df.groupby("label").mean()` shows whether a column differs between the answers you want to predict.
7. **Do the maths in bulk with numpy.** Operating on a whole array at once replaces a Python loop and is the habit every later post relies on.

## Why: the principle

**Look before you model.** A model learns whatever is in the data, including its mistakes. Missing values can break training or be silently filled in; duplicates can end up on both sides of a split, so the model is tested on rows it has already seen (Part 4); an imbalanced label makes a useless model look accurate. Each check above takes one line and rules out one of these.

**Scales differ.** The middle panel shows the largest value in each of the 30 columns: from 0.03 up to 4,254, a factor of more than 100,000. Many learning methods compare or add columns, and a column measured in thousands dominates one measured in hundredths unless you rescale. Part 5 does that.

**One column rarely tells the whole story, and some tell nothing.** The left panel: benign and malignant rows overlap between a mean radius of about 12 and 18, so no single threshold is perfect. The right panel shows how much each column knows alone, from 92% for the best down to 62.7% for the worst, which is exactly what always guessing "benign" scores. Combining many weak columns is what a model is for.

**Why numpy.** A numpy array stores numbers in one block and applies an operation to all of them in compiled code. The program below squares and adds a million integers in a Python loop and then in numpy: 60 ms against 3.7 ms on the machine used here, with the same answer. The absolute times depend on your machine; the gap does not. The libraries' own introductions are good references: [NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html) and [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html).

## Example, worked through

Reading the output below line by line:

- **Shape (569, 31):** 569 rows, 30 measurements and one label column. The program wrote the label as `benign`: 1 for benign, 0 for malignant.
- **No missing values and no duplicates.** Real data rarely arrives this clean; this dataset has been prepared for teaching.
- **Classes: 212 malignant, 357 benign.** 37% malignant, so a baseline that always says benign scores 62.7%. Any model has to beat that.
- **Ranges:** mean radius runs from 7 to 28, mean area from 143 to 2,501, mean smoothness from 0.05 to 0.16: three scales within three columns.
- **Class averages:** malignant rows average a mean radius of 17.5 against 12.1 for benign, and a mean area of 978 against 463. Smoothness differs much less (0.103 against 0.092), a hint that it is a weaker signal.
- **Single-column ranking:** the best columns ("worst radius", "worst area", "worst perimeter") reach 92% with one threshold; the weakest two sit at the 62.7% baseline, meaning they add nothing alone. The scores are computed on all rows with the threshold chosen on those same rows, so they are optimistic; Part 4 shows why that matters.

## Implementation

Save as `part2.py` and run it. It writes `tumours.csv` in the current folder and reads it back, which is the pattern for any file you will meet.

```python
import time
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

# 1. numpy: one operation on a whole array instead of a Python loop
values = list(range(1_000_000)); arr = np.array(values, dtype=np.int64)
t = time.perf_counter()
total_loop = 0
for v in values:
    total_loop += v * v
t_loop = time.perf_counter() - t
t = time.perf_counter(); total_np = int((arr * arr).sum()); t_np = time.perf_counter() - t
print(f"sum of squares of 1,000,000 numbers: loop {1000*t_loop:.0f} ms, numpy {1000*t_np:.1f} ms, same answer: {total_loop == total_np}")

# 2. pandas: a table with named columns. Save the dataset as a CSV file, then read it back as you would any real file.
bunch = load_breast_cancer(as_frame=True)
bunch.frame.rename(columns={"target": "benign"}).to_csv("tumours.csv", index=False)
df = pd.read_csv("tumours.csv")
print("\nshape:", df.shape)
print("first row of three columns:\n", df[["mean radius", "mean area", "mean smoothness"]].head(1).to_string(index=False))
print("\nmissing values in total:", int(df.isna().sum().sum()), "  duplicate rows:", int(df.duplicated().sum()))
print("rows per class (0 = malignant, 1 = benign):", df["benign"].value_counts().sort_index().to_dict())

# 3. the ranges of the columns, and how the two classes differ
desc = df[["mean radius", "mean area", "mean smoothness"]].describe().loc[["min", "max"]]
print("\nsmallest and largest value per column:\n", desc.round(3).to_string())
means = df.groupby("benign")[["mean radius", "mean area", "mean smoothness"]].mean().round(3)
print("\naverage per class:\n", means.to_string())

# 4. how well does one column alone separate the classes? best single threshold on each column
def best_threshold_accuracy(col):
    x, y = df[col].to_numpy(), df["benign"].to_numpy(); best = 0
    for th in np.unique(x):
        acc = max(((x > th) == y).mean(), ((x <= th) == y).mean()); best = max(best, acc)
    return best
cols = [c for c in df.columns if c != "benign"]
rank = sorted(((best_threshold_accuracy(c), c) for c in cols), reverse=True)
print("\nbest single-column threshold accuracy (on all rows, so optimistic):")
for acc, c in rank[:3] + rank[-2:]: print(f"  {c:25s} {100*acc:5.1f}%")
```

Output:

```
sum of squares of 1,000,000 numbers: loop 60 ms, numpy 3.7 ms, same answer: True

shape: (569, 31)
first row of three columns:
  mean radius  mean area  mean smoothness
       17.99     1001.0           0.1184

missing values in total: 0   duplicate rows: 0
rows per class (0 = malignant, 1 = benign): {0: 212, 1: 357}

smallest and largest value per column:
      mean radius  mean area  mean smoothness
min        6.981      143.5            0.053
max       28.110     2501.0            0.163

average per class:
         mean radius  mean area  mean smoothness
benign                                         
0            17.463    978.376            0.103
1            12.147    462.790            0.092

best single-column threshold accuracy (on all rows, so optimistic):
  worst radius               92.3%
  worst area                 92.1%
  worst perimeter            91.9%
  texture error              62.7%
  fractal dimension error    62.7%
```

## Try it

1. Open `tumours.csv` in a spreadsheet and check that `df.shape` matches what you see.
2. Add `df.isna().sum()` for every column rather than the total, then set a value to blank in the CSV and watch the count change.
3. Print the correlation between "mean radius" and "mean area" with `df["mean radius"].corr(df["mean area"])`. What does a value near 1 say about having both columns?
