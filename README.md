# citation-prediction

>Link Prediction in Citation Networks using Spark and Scala

## Supervised Learning Results

| Algorithm | Train time | Test time | F1 Score |
| - | :-: | :-: | -:|
| LSVC | 78sec | 86.85% | 1sec |
| Logistic Regression | 16sec | 87.61% | 2sec |
| Neural Network | 555sec | 71.45% | 3sec |
| Random Forest | 162sec | 90.8% | 6sec |
| GBoost Trees | 446sec | 92.75% | 7sec |

## Unsupervised Learning Results

| Algorithm | Train time | Test time | F1 Score |
| - | :-: | :-: | -:|
| k-means | 12sec | 30.61% | 1sec |
| Bis. k-means (Hierarchical clustering) | 42sec | 61.64% | 1sec |
| Gaussian Mixtures | 199sec | 45.72% | 2sec |

---
>_Machine Used: i7-3540 3GHz, 8GB RAM_