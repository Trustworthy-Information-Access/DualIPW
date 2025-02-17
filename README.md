# Unbiased Learning to Rank with Query-Level Click Propensity Estimation: Beyond Pointwise Observation and Relevance
This repo implements the source code of our paper *"Unbiased Learning to Rank with Query-Level Click Propensity Estimation: Beyond Pointwise Observation and Relevance"*, accepted by The WebConf (WWW) 2025. It is built on the [ULTRA toolbox](https://github.com/ULTR-Community/ULTRA_pytorch).

## Dataset
We used a subset of the BAIDU dataset, with the session IDs listed in the *sids.txt* file.

## Usage

**Run different ULTR methods**
Modify the parameter *method_name* to run the corresponding ULTR method, and make sure to pay attention to the parameters used by each method.
```
sh run.sh
```
Inverse Propensity Weighting (IPW) and Propensity Ratio Scoring (PRS) methods require the estimation of position bias, which we leverage All Pairs implemented in [Baidu-ULTR Reproducibility](https://github.com/philipphager/ultr-reproducibility) to estimate on our dataset.

Unconfounded Propensity Estimation (UPE) method requires the logging policy estimation, and we approximate this policy by training a LambdaMART model to fit the document positions.
```
sh gen_policy.sh
```