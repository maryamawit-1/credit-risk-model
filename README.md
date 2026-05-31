# credit-risk-model
# Credit Scoring Business Understanding

## 1. Basel II and Model Interpretability
Basel II requires financial institutions to use models that are transparent, well-documented, and explainable. This ensures that loan decisions can be justified to regulators and customers. As a result, credit scoring models must prioritize interpretability and stability over purely complex black-box performance.

## 2. Proxy Variable for Default
The dataset does not contain an explicit default label. Therefore, we construct a proxy variable using customer behavioral patterns (RFM analysis and clustering). Customers with low engagement and low monetary activity are treated as high-risk (proxy default = 1). This approach introduces assumptions and potential bias because the proxy may not perfectly reflect real repayment behavior.

## 3. Trade-offs Between Simple and Complex Models
Simple models such as Logistic Regression are highly interpretable and preferred in regulated environments, but may have lower predictive performance. Complex models like Random Forest or Gradient Boosting provide higher accuracy but are harder to interpret. In credit risk settings, interpretability and regulatory compliance are often more important than marginal gains in accuracy.