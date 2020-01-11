# 5200-web-mining

To reproduce training run and show current and historic KPIs:

```bash
$ dvc repro

$ dvc metrics show -a -T

$ dvc metrics show -a -T --type json --xpath validation.PearsonCorrelation
```