# 5200-web-mining

To reproduce training run and show current and historic KPIs:

```bash
$ dvc repro

# or w/o local training of current branch
$ python metrics.py
```

To view detailed metadata of experiments run:

```bash
$ dvc metrics show -a -T
```