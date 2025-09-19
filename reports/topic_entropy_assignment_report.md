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

**Chi-squared Test (Association with Dominant Topic):**
*   **COP26** stands out with **no significant association** for 'Reliability' and 'Political Leaning', both showing weak to moderate effect sizes. In contrast, for both the **COVID** and **Ukraine** datasets, all four variables ('Reliability', 'Political Leaning', 'Individual/Organization', and 'Category') showed **significant associations** with dominant topic, all with moderate effect sizes. This suggests that user characteristics were more consistently associated with their dominant topic in the COVID and Ukraine contexts than in COP26.

_____________________________________________

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


**Mann-Whitney U Test (Differences in Topic Entropy):**
*   **Reliability:** In **COP26** and **Ukraine**, no significant difference in topic entropy was found between reliable and questionable users. However, in the **COVID** dataset, reliable users' topic entropy **stochastically dominated** questionable users' topic entropy, indicating a likely difference.
*   **Political Leaning:**
    *   All three datasets showed **no significant two-sided differences** between R, L, and C leaning users among themselves (e.g., R vs L, R vs C, C vs L), with some exceptions and one-sided findings.
    *   The 'Unknown' political leaning group often showed **significant differences** when compared to other leanings. In **COP26**, R, C, and L leanings all stochastically dominated 'Unknown'. In **COVID**, C-leaning users stochastically dominated 'Unknown'. In **Ukraine**, L-leaning users' topic entropy was stochastically dominated by 'Unknown'. This suggests the 'Unknown' category might represent a group with distinct topic engagement patterns across the datasets.
*   **Individual vs. Organization:** In both **COP26** and **COVID**, individual users' topic entropy was **stochastically dominated** by organisation users, indicating that organisations tended to have higher topic entropy (i.e., less focused on a single topic). The **Ukraine** dataset also showed this one-sided stochastic dominance, although the two-sided test for overall difference was not significant (p=0.060). This implies a consistent pattern where individual users are more focused in their topic engagement than organisations across the board.



Category tests not present!

### COP26 Dataset

**Chi-squared Test and Cramer's V (Association with Dominant Topic)**
*   **Reliability:** The association was **not significant** (p=0.735) and was categorised as **weak** (Cramer's V=0.184).
*   **Political Leaning:** This association was also **not significant** (p=0.119), but it showed a **moderate** effect size (Cramer's V=0.218).
*   **Individual/Organization:** A **significant** association was found (p=0.008), with a **moderate** effect size (Cramer's V=0.271).
*   **Category:** A **highly significant** association was observed (p=4.81e-05), also with a **moderate** effect size (Cramer's V=0.252).

**Mann-Whitney U Test (Difference in Topic Entropy Distribution)**
*   **Reliability (Reliable vs. Questionable):** There was **no significant difference** in topic entropy between reliable and questionable users (p=0.978).
*   **Political Leaning (various comparisons):**
    *   No significant differences were detected between R vs L (p=0.562), R vs C (p=0.728), or C vs L (p=0.226) leaning users.
    *   However, **likely differences** were found when comparing specific leanings to 'Unknown' users:
        *   **R vs Unknown:** R-leaning users' topic entropy **stochastically dominates** that of Unknown users (p=0.0036).
        *   **C vs Unknown:** C-leaning users' topic entropy **stochastically dominates** that of Unknown users (p=0.00000).
        *   **L vs Unknown:** L-leaning users' topic entropy **stochastically dominates** that of Unknown users (p=0.00003).
*   **Individual vs. Organization:** A **likely difference** was observed (p=0.00018), with individual users' topic entropy being **stochastically dominated** by that of organisation users.

### COVID Dataset

**Chi-squared Test and Cramer's V (Association with Dominant Topic)**
*   **Reliability:** A **significant** association was found (p=3.46e-12), with a **moderate** effect size (Cramer's V=0.417).
*   **Political Leaning:** This also showed a **significant** association (p=1.83e-21) and a **moderate** effect size (Cramer's V=0.431).
*   **Individual/Organization:** A **significant** association was present (p=2.74e-17), with a **moderate** effect size (Cramer's V=0.504).
*   **Category:** A **highly significant** association was identified (p=4.83e-32), exhibiting a **moderate** effect size (Cramer's V=0.426).

**Mann-Whitney U Test (Difference in Topic Entropy Distribution)**
*   **Reliability (Reliable vs. Questionable):** A **likely difference** was found (p=0.01095), where reliable users' topic entropy **stochastically dominates** that of questionable users.
*   **Political Leaning (various comparisons):**
    *   No significant differences were detected between R vs L (p=0.802), R vs Unknown (p=0.424), or L vs Unknown (p=0.383) leaning users.
    *   **Likely differences** were observed in other comparisons:
        *   **R vs C:** R-leaning users' topic entropy is **stochastically dominated** by C-leaning users (p=0.00000).
        *   **C vs L:** C-leaning users' topic entropy **stochastically dominates** L-leaning users (p=0.00000).
        *   **C vs Unknown:** C-leaning users' topic entropy **stochastically dominates** Unknown users (p=0.00000).
*   **Individual vs. Organization:** A **likely difference** was detected (p=0.00000), with individual users' topic entropy being **stochastically dominated** by that of organisation users.

### Ukraine Dataset

**Chi-squared Test and Cramer's V (Association with Dominant Topic)**
*   **Reliability:** A **significant** association was found (p=0.0013), with a **moderate** effect size (Cramer's V=0.519).
*   **Political Leaning:** This also showed a **significant** association (p=0.00046) and a **moderate** effect size (Cramer's V=0.510).
*   **Individual/Organization:** A **significant** association was present (p=0.0033), with a **moderate** effect size (Cramer's V=0.541).
*   **Category:** A **highly significant** association was identified (p=1.57e-07), exhibiting a **moderate** effect size (Cramer's V=0.526).

**Mann-Whitney U Test (Difference in Topic Entropy Distribution)**
*   **Reliability (Reliable vs. Questionable):** There was **no significant difference** in topic entropy between reliable and questionable users (p=0.933).
*   **Political Leaning (various comparisons):**
    *   No significant differences were detected between R vs L (p=0.250), R vs C (p=0.323), R vs Unknown (p=0.150), or C vs Unknown (p=0.848) leaning users based on the two-sided test.
    *   For **C vs L**, the two-sided test showed no significant difference (p=0.068), however, a one-sided test indicated that C-leaning users' topic entropy **stochastically dominates** L-leaning users'.
    *   For **L vs Unknown**, a **likely difference** was found (p=0.01768), with L-leaning users' topic entropy being **stochastically dominated** by that of Unknown users.
*   **Individual vs. Organization:** The two-sided test showed **no significant difference** (p=0.060) in topic entropy between individual and organisation users. However, a one-sided test indicated that individual users' topic entropy is **stochastically dominated** by that of organisation users.

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
