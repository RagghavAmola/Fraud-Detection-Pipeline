# Evaluation Report — 2026-11-17

> **Note:** These figures describe one synthetic dataset generated on this date.
> Ground-truth (`is_fraud_actual`) is used here **only** for development validation.
> In production, this column would not exist.

## Dataset Summary

| Metric | Value |
|--------|-------|
| Total rows | 50,000 |
| Fraud rows | 1,400 (2.80%) |
| Normal rows | 48,600 |
| Tier counts | Medium: 635, High: 422, Critical: 44, Low: 48,899, Total: 50,000 |

## Threshold Evaluation

### Medium and above
Tiers included: Critical, High, Medium

**Confusion Matrix**

| | Predicted Normal | Predicted Fraud |
|---|---|---|
| **Actual Normal** | 48,285 (TN) | 315 (FP) |
| **Actual Fraud** | 614 (FN) | 786 (TP) |

| Metric | Value |
|--------|-------|
| Precision | 71.39% |
| Recall | 56.14% |
| F1 Score | 62.85% |
| False-Positive Rate | 0.65% |
| Normal rows flagged | 315 |

### High and above
Tiers included: Critical, High

**Confusion Matrix**

| | Predicted Normal | Predicted Fraud |
|---|---|---|
| **Actual Normal** | 48,546 (TN) | 54 (FP) |
| **Actual Fraud** | 988 (FN) | 412 (TP) |

| Metric | Value |
|--------|-------|
| Precision | 88.41% |
| Recall | 29.43% |
| F1 Score | 44.16% |
| False-Positive Rate | 0.11% |
| Normal rows flagged | 54 |

### Critical only
Tiers included: Critical

**Confusion Matrix**

| | Predicted Normal | Predicted Fraud |
|---|---|---|
| **Actual Normal** | 48,600 (TN) | 0 (FP) |
| **Actual Fraud** | 1,356 (FN) | 44 (TP) |

| Metric | Value |
|--------|-------|
| Precision | 100.00% |
| Recall | 3.14% |
| F1 Score | 6.09% |
| False-Positive Rate | 0.00% |
| Normal rows flagged | 0 |

## Per-Rule Analysis

| Rule | Normal Fired | Fraud Fired | Precision | Recall |
|------|-------------|-------------|-----------|--------|
| high_velocity | 0 | 120 | 100.00% | 8.57% |
| large_amount_deviation | 272 | 405 | 59.82% | 28.93% |
| odd_hour | 705 | 429 | 37.83% | 30.64% |
| high_risk_merchant | 0 | 200 | 100.00% | 14.29% |
| odd_hour_high_risk_merchant | 0 | 33 | 100.00% | 2.36% |
| new_merchant_large_amount | 212 | 440 | 67.48% | 31.43% |
| foreign_high_risk_country | 26 | 54 | 67.50% | 3.86% |
| foreign_medium_risk_country | 135 | 59 | 30.41% | 4.21% |
| foreign_low_risk_country | 155 | 77 | 33.19% | 5.50% |

## Limitations

- **Synthetic data**: all transactions are generated, not real.
- **No legitimate high-risk-merchant purchases**: the `high_risk_merchant` rule fires on 0 normal rows because the generator only places high-risk merchant categories in fraud-injected transactions.
- **No legitimate rapid bursts**: all rapid-fire (5 txns in 5 min) patterns are injected as fraud; real data would contain some legitimate bursts.
- **Amount distribution narrower than real data**: per-card amounts are drawn from a single log-normal; real spending has heavier tails and seasonal variation.
- **Warm-up rows are neutral by design**: the first 5 transactions per card establish the home country and are never fraud-injected, inflating TN count.
- **Grey-list travel is the only legitimate high-tier activity**: ~4% of cards get one travel episode to a high-tier grey-list country; this is the sole source of false positives for the `foreign_high_risk_country` rule.
