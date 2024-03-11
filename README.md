# My Recommendation study zone

### Evaluate Matric

#### Ranking Matric
#### NDCG(Normalized Discounted Cumulative Gain) : 추천된 Top K개를 추천 순위에 가중치를 두어 평가한다. 
NDCG : DCG를 정규화 한것
DCG : relavence의 가중합, 이때 가중치는 내가 예측한 순위의 역수

$$\displaystyle CG_k = \sum_{i=1}^{k} rel_{i}$$
rel은 사용자가 아이템(i)를 얼마나 선호하는지에 대한 값으로, 보통 평점을 사용
$$\displaystyle DCG_k = \sum_{i=1}^{k} \frac{rel_{i}}{log_2(i+1)}$$

#### HR(Hit ratio) : 전체 사용자수 대비 적중한 사용자 수
1. 사용자가 선호한 아이템 1개를 제외하고 학습
2. 사용자 별로 k개 아이템을 추천하고, 제외한 아이템이 포함되면 Hit
3. Hit된 사용자 수 / 전체 사용자 수 = Hit Ratio

#### MRR(Mean reciprocal rank)
### Prediction Matric
#### MAE & MSE

