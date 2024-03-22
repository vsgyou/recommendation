# Invariant Preference Learning for General Debiaseing in Recommendation

## Notation

>- U : user node 
>- V : item node 
>- M : invariant true preference of user for item 
>- A : variant user preference for the item affected by latent confounder 
>- Y : user feedback on the item in the observational data 
>- C : the set of confounders that my cause biases in the observational data 
>- E : the enviroment which is the agents for C

## Propose

>- The heterogeneity of environment can regared to be caused by confounders.
As a result, we can directly use it as the agents of confounders

>- we propose a novel Invariant Preference Learning framework according to the causal graph.
##### Problem 1 (General Debiaseing Problem)
>- Gevin heterogeneous training data $D = D_{e}$ collected from multiple environments $e \in \epsilon$ without explicit labels, the tast is to exploit the latent heterogeneity inside data and capture the invariant preference for general debiasing

## InvPref
#### Environment Inference
>- Propose to generate environments with a clustering algorithm based on $p_{e}(y|u,v)$, given a $(u,v)$ pair, we infer its feedback $\hat{y}_{u,v,e}$ under each environment $e$ and select the environment $\hat{e}_{u,v}$ corresponding to the result closest to $y_{u,v}$, which is the true feedback of $(u,v)$
#### Invariant Preference Learning
>- Given the learned envireonments, we capture invariant and variant preference via adversarial learning. we use different embedings to capture the invariant preference, variant preference and latent environments respectively. To learn discriminative and invariant true user preference across multiple environments, InvPref jointly optimizes the recommendation task and an environment classifier which is only used in the trining phase.