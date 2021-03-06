# citation-prediction

> Link Prediction in Citation Networks using Spark and Scala

[![license](https://flat.badgen.net/github/license/iamnapo/citation-prediction)](./LICENSE)

## Supervised Learning Results

|      Algorithm      | Train time | Test time |  F1 Score  |                                             Hyper-Parameters                                             |
| :-----------------: | :--------: | :-------: | :--------: | :------------------------------------------------------------------------------------------------------: |
|        LSVC         |   102sec   |   6sec    |   87.4%    | <ul><li>maxIter: 2000</li><li>regParam: 0.001</li><li>tol: 1.0E-8</li><li>aggregationDepth: 10</li></ul> |
| Logistic Regression |   16sec    |   3sec    |   87.9%    |                           <ul><li>maxIter: 2000</li><li>tol: 1.0E-8</li></ul>                            |
|   Neural Network    |   201sec   |   4sec    |   73.67%   |                                <ul><li>layers: [14, 32, 18, 2]</li></ul>                                 |
|    Random Forest    |   76sec    |   17sec   |   93.59%   |                                      <ul><li>maxDepth: 15</li></ul>                                      |
|  **GBoost Trees**   |   681sec   |   53sec   | **95.92%** |                                      <ul><li>maxDepth: 15</li></ul>                                      |

## Unsupervised Learning Results

|               Algorithm                | Train time | Test time |  F1 Score  |                  Hyper-Parameters                   |
| :------------------------------------: | :--------: | :-------: | :--------: | :-------------------------------------------------: |
|                k-means                 |    9sec    |   1sec    |   76.17%   | <ul><li>maxIter: 1000</li><li>tol: 1.0E-9</li></ul> |
| Bis. k-means (Hierarchical clustering) |   26sec    |   1sec    |   33.59%   |            <ul><li>maxIter: 50</li></ul>            |
|           Gaussian Mixtures            |   49sec    |   1sec    |   2.95%    | <ul><li>maxIter: 1000</li><li>tol: 1.0E-9</li></ul> |
|         **Outlier detection**          |    1sec    |   2sec    | **86.73%** |                     <ul>-</ul>                      |

---

> _Machine: i7-3615QM @ 2.3GHz, 8GB RAM_

## License

AGPL-3.0 © [Napoleon-Christos Oikonomou](https://iamnapo.me)
