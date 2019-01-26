# citation-prediction

>Link Prediction in Citation Networks using Spark and Scala

## Supervised Learning Results

| Algorithm | Train time | Test time | F1 Score | Hyper-Parameters |
| :-: | :-: | :-: | :-: | :-: | :-: |
| LSVC | 102sec | 6sec | 87.4% | <ul><li>maxIter: 2000</li><li>regParam: 0.001</li><li>tol: 1.0E-8</li><li>aggregationDepth: 10</li></ul> |
| Logistic Regression | 16sec | 3sec | 87.9% | <ul><li>maxIter: 2000</li><li>tol: 1.0E-8</li></ul> |
| Neural Network | 201sec | 4sec | 73.67% | <ul><li>layers: [14, 32, 18, 2]</li></ul> |
| Random Forest | 76sec | 17sec | 93.59% | <ul><li>maxDepth: 15</li></ul> |
| __GBoost Trees__ | 681sec | 53sec | __95.92%__ | <ul><li>maxDepth: 12</li></ul> |

## Unsupervised Learning Results

| Algorithm | Train time | Test time | F1 Score | Precision of citation | Hyper-Parameters |
| :-: | :-: | :-: | :-: | :-: | :-: |
| k-means | 9sec | 1sec | 76.17% | 0.37%| <ul><li>maxIter: 1000</li><li>tol: 1.0E-9</li></ul> |
| Bis. k-means (Hierarchical clustering) | 26sec | 1sec | 33.59% | 0.2% | <ul><li>maxIter: 50</li></ul> |
| Gaussian Mixtures | 49sec | 1sec | 2.95% | 0.07% | <ul><li>maxIter: 1000</li><li>tol: 1.0E-9</li></ul> |
| __Outlier detection__ | 1sec | 2sec | __86.73%__ | 0.63% | <ul>-</ul> |
---
>_Machine: i7-3615QM @ 2.3GHz, 8GB RAM_