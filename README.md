# Predicting Purchase Quantity in Online Retail Transactions

A Data Cleaning, Feature Engineering, and Linear Regression Analysis. Submitted for DAMO-510-8 (Predictive Analytics), Data Analytics MSc, University of Niagara Falls. Instructor: Prof. Payman Janbakhsh. Submitted August 9, 2026.

Student: Jose Alberto Martinez Morales.

## 1. Introduction

The project uses the Online Retail II dataset (Chen, 2012): transaction-level records from a UK-based online retailer of gift merchandise, spanning December 2009 to December 2011 across more than one million rows. Many customers are wholesalers, so the data is rich but messy: duplicate entries, cancellations, inconsistent product codes, and missing identifiers throughout.

The goal is a regression model that predicts purchase quantity in a completed transaction, useful for inventory planning and for flagging customers or products likely to generate large orders. Getting there required cleaning five main problems first: cross-worksheet duplicates, cancellations and administrative adjustments mixed into the transaction log, non-product stock codes, missing customer IDs on about a fifth of rows, and extreme, heavily right-skewed quantity and price values driven by genuine wholesale orders.

## 2. Data Cleaning

**Missing values.** Customer ID was missing in 22.6% of transactions. Rather than dropping these rows, the gap was kept as a signal: missing-ID transactions averaged a quantity of 3.04, against 13.52 for registered customers, so a binary flag preserved that difference instead of discarding it.

**Duplicates.** An exact match found 23,430 duplicate rows; a key-column match found 67,246, exposing 43,812 cross-year duplicates where the same purchase showed up in both source worksheets. Invoice number alone was never used to define a duplicate, since one invoice legitimately spans several line items. 34,337 duplicate rows were removed in total.

**Cancellations and returns.** Invoice numbers starting with "C" marked 19,104 cancellations (1.85%), and a separate 3,393 rows were negative-quantity returns without that prefix. Sorting everything into four categories produced 1,006,013 completed sales, 17,915 cancellations, 3,393 returns, and 5,713 adjustments.

**Negative and zero values.** All 22,496 negative-quantity rows were accounted for across cancellations, returns, and adjustments. Five negative-price rows traced back to bad-debt write-offs. Of 6,014 zero-price rows, 2,594 completed-sale rows were 97.7% missing a Customer ID or Description and got excluded as likely not genuine sales; 60 zero-price sales with real identifiers were kept.

**Unusual stock codes.** 62 non-digit stock codes (5,980 rows) were checked one by one against description, price, and customer coverage. That surfaced postage, discount, and bank-charge codes, plus a "channel transfer" pattern with zero customer-ID attachment despite product-like descriptions, which was excluded. Separately, 369 rows had write-off notes typed straight into the product description field.

**Outliers.** Quantity ran as high as 80,995 units, against a 99th percentile of 108. An IQR-based rule was tried and rejected: it flagged around 10% of rows, which doesn't make sense for a wholesale-heavy retailer. Extreme values were checked individually and kept rather than cut.

**Text standardization.** Whitespace was stripped from key text fields, and Description was standardized to uppercase using a dominant-description rule, with a colour-conflict check added so genuine product variants didn't get merged into one. Country names were standardized wherever the mapping was unambiguous.

**Final rules.** The cleaned dataset keeps only completed-sale rows classified as genuine merchandise, excluding administrative codes and non-genuine zero-price rows. That leaves 1,003,231 rows across 41 columns.

## 3. Data Transformation

**Date and time features.** InvoiceDate was broken into Year, Month, DayOfWeek, and Hour, plus derived Is_Weekend, Time_Of_Day, and Season flags, so the model can pick up seasonal, weekly, and time-of-day demand patterns directly.

**Encoding.** High-cardinality identifiers (StockCode, Customer ID, Description) were frequency-encoded instead of one-hot encoded, avoiding sparse columns with little signal. Low-cardinality variables (Country, DayOfWeek) were one-hot encoded, with countries under 100 transactions grouped into "Other."

**Box-Cox, Yeo-Johnson, and target transformation.** Quantity, Price, and the frequency encodings were extremely right-skewed, with skewness values from 2.5 to over 450. Box-Cox (Box & Cox, 1964) was applied to Quantity and Price; Yeo-Johnson (Yeo & Johnson, 2000) suited the frequency encodings, which can hit zero. Applying Box-Cox to the target (lambda = -0.2538, fit on training data only) raised R-squared from 0.025/-0.003 to 0.320/0.252. But once predictions were converted back to the raw scale, they carried more error than the original untransformed model, a retransformation-bias effect from the compressive negative-lambda transform.

**Feature engineering.** Expanding-window historical features (prior transaction counts, cumulative and average quantity per product, customer, and country) were built by sorting on InvoiceDate and excluding the current row. One bug turned up and got fixed: guest transactions initially shared a single placeholder ID, which falsely linked around 226,000 unrelated transactions into one continuous history. Historical features now reset to zero for guests.

**Interaction terms.** Three were built: Price × Product_Historical_TxnCount (price sensitivity), Is_Weekend × Hour (weekday versus weekend timing), and Customer_Historical_TxnCount × Product_Historical_TxnCount (loyalty and familiarity).

**Leakage prevention.** The train/test split was chronological, so the model never trains on future information, and historical features were confirmed to use only prior rows. One leakage risk remains open: the frequency encodings were supposed to be refit on the training set only but were instead computed on the full dataset before the split. That's a known limitation, covered in Section 5.

## 4. Regression Modeling

The final dataset: 1,003,231 completed merchandise transactions, 41 columns. After removing identifiers and redundant features flagged by multicollinearity checks, the feature set combined numeric predictors (price, frequency encodings, historical aggregates, interaction terms, calendar variables) with one-hot encoded categorical predictors (grouped country, day of week).

The split was chronological: the earliest 80% for training, the most recent 20% for test (seed = 42), to keep time order intact. That test window lands around September to December 2011, which skews toward the pre-holiday sales surge, an accepted trade-off given the chronological approach.

Numeric predictors were standardized using training-data statistics only. Variance Inflation Factor checks removed four redundant features; remaining VIF values stayed below 5 except for several country indicators, expected given one-hot encoding on a UK-dominated dataset.

| Metric | Training | Test |
|---|---|---|
| R-squared (raw target) | 0.025 | -0.003 |
| R-squared (Box-Cox target) | 0.320 | 0.252 |
| RMSE (raw target) | 109.23 | 184.55 |
| MAE (raw target) | 11.29 | 10.72 |

Because predictors were standardized, each coefficient reflects a one-standard-deviation change. Customer_Historical_TotalQty was the strongest predictor (+16.4): customers with more purchase history place larger orders. Customer_Historical_TxnCount came in negative (-6.8), meaning customers who transact often tend to place smaller individual orders. Price carried a modest negative coefficient, matching basic demand logic. Country coefficients varied a lot; Denmark's (+337) likely reflects a handful of unusually large orders rather than a stable pattern. All of this describes association, not cause and effect.

Diagnostics ruled out multicollinearity, but linearity, homoscedasticity, and normality were each violated, along with a mild independence violation (Durbin-Watson = 1.56), driven mostly by a small number of extreme-quantity transactions. The size of the improvement from the Box-Cox correction points to target skewness, not predictor choice, as the main cause of the raw-scale model's weak fit.

## 5. Evaluation and Reflection

**Main challenges.** The hardest problems were about data quality, not modeling: telling genuine sales apart from cancellations and administrative noise, catching cross-year duplication that exact-matching alone would miss, and building historical features that stay leakage-free once guest customers are handled correctly.

**Effects of cleaning decisions.** Keeping extreme-quantity transactions instead of cutting them with an IQR rule is the main reason for the diagnostic violations in Section 4. Encoding a missing Customer ID as its own feature preserved a real behavioral difference between guest and registered customers, though it took a later fix once its distorting effect on historical features surfaced.

**Data-leakage risks.** The chronological split and the historical features check out as leakage-free. The frequency encodings don't: they were computed on the full dataset before the split existed, so the reported test-set numbers are probably a bit more optimistic than a fully leakage-free version would show.

**Model limitations.** Quantity is positive, discrete, and heavily right-skewed, which runs against core Linear Regression assumptions and shows up as negative predictions and non-normal residuals on the raw target. The Box-Cox transform improved the statistical fit but introduced retransformation bias in exchange. Most of the model's explanatory power comes from a small number of customer-history features.

**Recommendations.** Future work should recompute the frequency encodings on the training set only, apply a bias-correction method such as Duan's (1983) smearing estimator for raw-scale predictions, and consider Poisson or Negative Binomial GLMs, which fit positive, discrete, overdispersed count data better than standard Linear Regression.

## References

Box, G., & Cox, D. (1964). An analysis of transformations. *Journal of the Royal Statistical Society: Series B, 26*(2), 211–252.

Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D

Chen, D., Sain, S., & Guo, K. (2012). Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining. *Journal of Database Marketing & Customer Strategy Management, 19*(3), 197–208. https://doi.org/10.1057/dbm.2012.17

Duan, N. (1983). Smearing estimate: A nonparametric retransformation method. *Journal of the American Statistical Association, 78*(383), 605–610. https://doi.org/10.1080/01621459.1983.10478017

Yeo, I., & Johnson, R. (2000). A new family of power transformations to improve normality or symmetry. *Biometrika, 87*(4), 954–959. https://doi.org/10.1093/biomet/87.4.954
