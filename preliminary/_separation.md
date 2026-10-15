| features | BIC, 1 cluster | BIC, 2 clusters | BIC, 3 clusters | preferred | ARI of 2 clusters vs tags |
|:--|--:|--:|--:|--:|--:|
| body volume + dendrite count | 812.0 | 833.3 | 839.4 | 1 | 0.24 |
| body volume + dendrite count + cell axis ratio | 1243.0 | 1255.3 | 1279.0 | 1 | 0.30 |

: Gaussian mixtures on standardised log body volume, log(1 + dendrite count) and axis ratio; lower BIC is better; ARI 1 = clusters identical to the tags, 0 = chance. {#tbl-mixture}

| features | AUC (leave one stack out) | 95% CI |
|:--|--:|--:|
| body volume | 0.72 | 0.63 – 0.81 |
| dendrite count | 0.76 | 0.67 – 0.84 |
| body volume + dendrite count | 0.76 | 0.67 – 0.85 |
| cell axis ratio | 0.79 | 0.71 – 0.86 |
| body thickness | 0.79 | 0.71 – 0.87 |
| body volume + dendrite count + cell axis ratio | 0.85 | 0.78 – 0.91 |

: Agreement of a logistic regression with the tags, each stack predicted by a model trained on the others; AUC 0.5 = chance, 1 = perfect. 95% CI by bootstrap over cells. {#tbl-auc}
