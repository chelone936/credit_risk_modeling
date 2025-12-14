## Credit Scoring Business Understanding

### How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

According to the Basel II Capital Accord, banks are required to hold capital based on how risky their loans are, rather than using a fixed rule for all loans. Basel II allows banks to use internal models to estimate credit risk measures such as Probability of Default (PD), as long as those models meet strict regulatory standards (Basel Committee on Banking Supervision).

Because these model outputs directly affect how much capital a bank must hold, regulators must be able to understand, review, and trust the model. This means the model must be easy to explain and well documented. According to the Basel II framework, models should be transparent and open to validation and supervisory review.

As a result, interpretable models are preferred in regulated environments. Models such as logistic regression allow risk teams and regulators to clearly see how each variable affects the final credit score. This helps ensure compliance, supports audits, and reduces regulatory risk. If a model cannot be explained clearly, regulators may reject it or require the bank to hold extra capital.

---

### Since we lack a direct “default” label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

In credit risk modeling, the ideal target variable is whether a borrower actually defaulted on a loan. However, in many datasets, this information is not directly available. According to the World Bank’s credit scoring guidelines, when a true default label is missing, a proxy variable must be created using related signals such as missed payments, long delays, or other signs of financial trouble.

A proxy variable is necessary because supervised models need a clear outcome to learn from. Without a target that represents bad credit behavior, the model cannot learn how to separate low-risk and high-risk borrowers.

However, using a proxy comes with risks. According to the World Bank, if the proxy does not closely match real default behavior, the model may produce misleading results. This can lead to:

- Approving risky borrowers who later default  
- Rejecting good borrowers unfairly  
- Incorrect pricing of loans  
- Increased financial losses  

There is also regulatory risk. If the proxy is poorly defined or not well explained, regulators may question the reliability of the model. For these reasons, the proxy definition must be clearly justified, tested, and documented.

---

### What are the key trade-offs between using a simple, interpretable model versus a complex, high-performance model in a regulated financial context?

According to credit risk modeling literature and industry practice, there is a clear trade-off between model simplicity and model performance.

Simple models, such as logistic regression with Weight of Evidence (WoE), are widely used in banking because they are easy to understand. Each variable has a clear and direct impact on the predicted risk. According to several credit risk studies, these models are easier to explain to regulators, auditors, and business users. They are also easier to validate and maintain over time.

More complex models, such as Gradient Boosting, usually perform better in terms of prediction accuracy. According to data science research, these models can capture non-linear relationships and interactions that simple models cannot. This often leads to better risk separation.

However, complex models are harder to explain. Even with explanation tools, their decisions may still be difficult for regulators and business teams to fully understand. This increases model governance effort and regulatory scrutiny.

In a regulated financial environment, banks must balance these factors. According to industry guidance, a slightly less accurate but well-understood model is often preferred over a highly accurate model that cannot be easily explained or defended.
