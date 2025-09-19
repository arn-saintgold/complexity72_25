# Topic Modeling and Topic Entropy

We applied BERTopic to the COP26, COVID, and Ukraine datasets, using transformer-based embeddings (all-mpnet-base-v2), dimensionality reduction, and density-based clustering to extract topics, optimizing for cluster quality using relative validity. 

Topic quality was evaluated with c_v coherence and two diversity measures: TopicDiversity and Word Embedding-based Diversity (WED). Dominant topics were assigned to users using frequency- and entropy-based strategies, excluding noise topics and applying thresholds to ensure focused assignment, with topic entropy and effective topic numbers capturing engagement diversity. 

Associations between dominant topics and user metadata (Reliability, Political Leaning, Individual/Organization, Category) were assessed via χ² tests, and Mann-Whitney U tests compared entropy distributions, revealing dataset-specific patterns of user-topic engagement.

## Differences Across Datasets

|   Data  | Reliability | Leaning | Ind/org | Category |
| ------- | ----------- | ------- |-------- | -------- |
|  Cop26  |     ✔       |    ✔    |    ✔    |     ✔    |
|  Covid  |     ✔       |    ✔    |    ✔    |     ✔    |
| Ukraine |     ✔       |    ✔    |    ✔    |     ✔    |

χ² tests between assigned user topics and metadata
✔ indicates significant and at least moderate association; 
✘ indicates non significant association or weak association.


|   Data  | R vs Q | Ind/Org |
| ------- | ------ | ------- |
|  Cop26  |        |    ✘    |
|  Covid  |   ✔    |    ✘    |
| Ukraine |        |    ✘    |

Mann-Whitney tests between topic entropy and Reliability, and Individual/Organization association. ✔ indicates significant stochastical dominance of the first class (R, Ind) over the second one (Q, Org), meaning that the first has more entropy than the second; ✘ indicates the reverse. No indication indicates no significant difference.

|   Data  |  R vs L | R vs C | C vs L | R vs ? | C vs ? | L vs ?|
| ------- |  ------ |------- | ------ | ------ | ------ | ----- |
|  Cop26  |         |        |        |   ✔    |    ✔   |   ✔   |
|  Covid  |         |    ✘   |    ✔   |        |    ✔   |       |
| Ukraine |         |        |    ✔   |        |        |   ✘   |

Mann-Whitney tests between topic entropy and Political leaning.


**Chi-squared Test (Association with Dominant Topic):**
*   For all the datasets, all four variables ('Reliability', 'Political Leaning', 'Individual/Organization', and 'Category') showed **significant associations** with dominant topic, all with moderate effect sizes. This suggests that user characteristics were more consistently associated with their dominant topic in the COVID and Ukraine contexts than in COP26.

_____________________________________________


**Mann-Whitney U Test (Differences in Topic Entropy):**
*   **Reliability:** In **COP26** and **Ukraine**, no significant difference in topic entropy was found between reliable and questionable users. However, in the **COVID** dataset, reliable users' topic entropy **stochastically dominated** questionable users' topic entropy, indicating a likely difference.
*   **Political Leaning:**
    *   All three datasets showed **no significant two-sided differences** between R, L, and C leaning users among themselves (e.g., R vs L, R vs C, C vs L), with some exceptions and one-sided findings.
    *   The 'Unknown' political leaning group often showed **significant differences** when compared to other leanings. In **COP26**, R, C, and L leanings all stochastically dominated 'Unknown'. In **COVID**, C-leaning users stochastically dominated 'Unknown'. In **Ukraine**, L-leaning users' topic entropy was stochastically dominated by 'Unknown'. This suggests the 'Unknown' category might represent a group with distinct topic engagement patterns across the datasets.
*   **Individual vs. Organization:** In both **COP26** and **COVID**, individual users' topic entropy was **stochastically dominated** by organisation users, indicating that organisations tended to have higher topic entropy (i.e., less focused on a single topic). The **Ukraine** dataset also showed this one-sided stochastic dominance, although the two-sided test for overall difference was not significant (p=0.060). This implies a consistent pattern where individual users are more focused in their topic engagement than organisations across the board.

***

### 1. Statistical Test for Association (Categorical Data)

The $\chi^2$ test, paired with Cramer's V to measure effect size, was used to determine if there was an association between a user's metadata (e.g., reliability, political leaning, category) and their dominant topic assignment. An effect size (ES) between 0.2 and 0.6 is interpreted as a moderate association.

#### Consistency Across Datasets (All Associations are Moderate and Significant)

Across all three datasets (COP26, COVID, and UKRAINE), the statistical results consistently showed that user metadata variables are **significantly and moderately associated** with the assigned dominant topic.

| Metadata Variable | COP26 (Cramer's V) | COVID (Cramer's V) | UKRAINE (Cramer's V) |
| :--- | :--- | :--- | :--- |
| **Reliability** | 0.461 | 0.417 | 0.519 |
| **Political Leaning** | 0.462 | 0.431 | 0.510 |
| **Individual/Organization** | 0.503 | 0.504 | 0.541 |
| **Category** | 0.482 | 0.426 | 0.526 |

For all comparisons across all datasets, the p-values were low (e.g., far less than 0.05), indicating **strong evidence** that topic assignment and these metadata fields are not independent. The Cramer's V values consistently fell within the **moderate association range** (0.2 < ES $\le$ 0.6).

***

### 2. Tests for Difference in Distribution (Topic Entropy)

The Mann-Whitney U test compares whether two independent distributions (in this case, topic entropy of different user groups) are equal in location and shape. A p-value less than 0.05 indicates a likely difference.

#### Differences based on Reliability

*   **COP26 & UKRAINE:** There was **no significant difference** in topic entropy detected between reliable and questionable users in the COP26 dataset (p-value: 0.52892) or the UKRAINE dataset (p-value: 0.93314).
*   **COVID:** A **likely difference** was found in the COVID dataset (p-value: 0.01095). Here, **reliable users' topic entropy stochastically dominates questionable users' entropy**.

#### Differences based on Individual vs. Organization Status

*   **COP26:** There was **no significant difference** in topic entropy between Individual and Organization users (p-value: 0.43853).
*   **UKRAINE:** There was **no significant difference** in the two-sided test (p-value: 0.06048). However, a one-sided test suggested that Individual users' topic entropy is stochastically dominated by Organization users' entropy.
*   **COVID:** A **likely difference** was detected (p-value: 0.00000). Individual users' topic entropy is **stochastically dominated by Organization's** entropy.

#### Differences based on Political Leaning

The Mann-Whitney U test revealed varying levels of difference based on political leaning (R=Right, L=Left, C=Centrist, Unknown) across the three datasets:

**A. Comparisons between R, L, and C:**

*   **R vs L, R vs C, C vs L (COP26 & UKRAINE):** No significant difference in topic entropy was detected between R, L, and C leaning users in the COP26 or UKRAINE datasets.
*   **R vs C (COVID):** A **likely difference** was found (p=0.00000), where R-leaning users' topic entropy is stochastically dominated by C's.
*   **C vs L (COVID):** A **likely difference** was found (p=0.00000), where C-leaning users' topic entropy stochastically dominates L's.
*   **C vs L (UKRAINE):** The two-sided test was not significant (p=0.06792), but a one-sided test indicated that C-leaning users' topic entropy stochastically dominates L's.

**B. Comparisons between Specific Leaning and Unknown:**

The greatest variation occurs when comparing users with specific leanings (R, C, L) against those with an "Unknown" leaning:

| Comparison | COP26 Result | COVID Result | UKRAINE Result |
| :--- | :--- | :--- | :--- |
| **R vs Unknown** | **Likely difference** (p=0.00099). R stochastically dominates Unknown's entropy. | No significant difference (p=0.42373). | No significant difference (p=0.14972). |
| **C vs Unknown** | **Likely difference** (p=6.96e-05). C stochastically dominates Unknown's entropy. | **Likely difference** (p=0.00000). C stochastically dominates Unknown's entropy. | No significant difference (p=0.84842). |
| **L vs Unknown** | **Likely difference** (p<0.05). L stochastically dominates Unknown's entropy. | No significant difference (p=0.38324). | **Likely difference** (p=0.01768). L is stochastically dominated by Unknown's entropy. |

In summary:

*   **COP26** showed that **all specified political leanings (R, C, and L)** had topic entropies that stochastically dominated the 'Unknown' group's entropy (i.e., known leanings tended to have *lower* entropy/higher focus).
*   **COVID** only showed a **likely difference for Centrist (C)** users versus Unknown, with C stochastically dominating Unknown.
*   **UKRAINE** showed a **likely difference only for Left (L)** users versus Unknown, but in this case, L's entropy was stochastically *dominated by* Unknown's (meaning L-leaning users had *lower* entropy than the Unknown group).
## Methodology (Heavy AI Help)

### Topic Model

We performed topic modeling on the three dataset using the BERTopic framework, which integrates transformer-based embeddings, dimensionality reduction, and density-based clustering into a unified pipeline. Texts were first cleaned removing empty strings, URLs, and extraneous phrases (HTML escape sequences, boilerplate text), while normalizing mentions and hashtags. Duplicate documents were removed, retaining the first occurrence, and the cleaned texts were stored alongside the original texts for reference.

Semantic embeddings were obtained using the `all-mpnet-base-v2` SentenceTransformer model. Precomputed embeddings were loaded when available, otherwise they were generated from the cleaned texts. The BERTopic pipeline then reduced the embeddings’ dimensionality, identified clusters, and extracted interpretable topics using class-based TF-IDF weighting of unigrams and bigrams. Optimal pipeline parameters, including dimensionality reduction and clustering settings as well as random seed, were selected by maximizing cluster quality measured through validity index. Each document was assigned to a topic, with low-density points labeled as noise.

### Topic Quality

The quality of the topic models was assessed using coherence and diversity metrics. Document texts were first preprocessed removing URLs, mentions, hashtags, emojis, and extraneous whitespace, followed by lemmatization. Topics generated by the BERTopic pipeline were similarly cleaned and lemmatized. 

Coherence was computed using the `c_v` coherence metric, which evaluates the semantic similarity among the top words within each topic. Diversity was measured using two complementary approaches. First, a standard TopicDiversity metric quantified the proportion of unique terms across topics, reflecting the distinctiveness of topic representations. Second, Word Embedding-based Diversity (WED) was calculated by mapping topic words to pre-trained word embeddings (Word2Vec) and computing the average pairwise cosine distance between all words within each topic, providing a semantic measure of topic spread. These metrics were applied consistently across all datasets to provide standardized, interpretable measures of topic model quality.

| Dataset |  N Docs | Min Clust Size |N Topics | Noise % |   RV   |   C_V  | Diversity |   WED  |
| ------- | ------- | -------------- | ------- | ------- | ------ | ------ | --------- | ------ | 
|  Cop26  | 105,383 |       150      |    77   |  39.65% | 40.40% | 52.13% |   72.18%  | 80.38% | 
|  Covid  | 545,032 |       250      |   163   |  41.74% | 34.75% | 47.33% |   66.55%  | 88.86% |
| Ukraine | 787,872 |       500      |   137   |  48.61% | 32.35% | 54.37% |   71.80%  | 76.82% |


### Topic Entropy and Assignment


The assignment of topics to users was performed through a structured methodology combining data preparation and defined assignment strategies with explicit thresholds. Cleaned tweet data were first merged with tweet-topic assignments, and user metadata—including political stance, reliability, individual/organization status, and user category—was subsequently integrated, linking user characteristics to their content engagement.

Two primary strategies were used to assign a dominant topic to each user: a frequency-based approach and an entropy-based approach. In both cases, noise topics were excluded from all calculations, and users who did not meet the criteria of a given strategy were not assigned a dominant topic.

The frequency-based approach assigned a dominant topic to each user based on the most frequently occurring topic among their tweets. A topic was assigned if its frequency exceeded a predefined threshold, with ties resolved by selecting the topic with the lowest numerical ID. This approach offered straightforward interpretability and generally resulted in higher user retention.

The entropy-based approach aimed to assign topics only when a user’s topic distribution indicated sufficient focus. For each user, all associated topics (excluding noise) were compiled, and topic entropy was calculated using a standard entropy function. Topic entropy quantified the diversity of a user’s topic engagement: values of 0 indicated perfect focus on a single topic, ~1 reflected dominance across two topics, ~2 corresponded to roughly four topics, and ~3 indicated engagement across approximately eight topics.
A dominant topic was assigned only if the user’s topic entropy fell below a specified maximum threshold, and assignments were explicitly prevented in cases of ties at this threshold. Although this method resulted in lower user retention due to its stricter focus criterion, it provided a more robust measure of individual topical focus.

Finally, the results from both assignment strategies, including topic entropy and effective topic numbers, were combined with user metadata for downstream analyses. User retention rates, defined as the percentage of users receiving non-null topic assignments, were evaluated across different threshold settings to inform the selection of the assignment methodology.

To evaluate differences in association with a dominant topic we performed χ² tests on topics and users thereby associated.
