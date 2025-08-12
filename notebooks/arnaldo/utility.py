from math import log2
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import seaborn as sns

# Topic Assignment
def test():
    print("I am here!")

def get_user_frequent_topic(topics, threshold=0.5, exclude=-1):
    """
    Args:
        topics (list): list of topic labels for a user.
        threshold (float): minimum fraction (e.g., 0.5 for 50%) for frequence.
        exclude (int or list): topic(s) to exclude, e.g. -1 for bertopic Noise.
    
    Returns:
        int or None: dominant topic if above threshold, else None
    """
    if isinstance(exclude, int):
        topics = [t for t in topics if t != exclude]
    elif isinstance(exclude, list):
        topics = [t for t in topics if t not in exclude]

    if not topics:
        return None

    total = len(topics)
    topic_counts = Counter(topics)
    top_topic, top_count = topic_counts.most_common(1)[0]
    
    if top_count / total >= threshold:
        return int(top_topic)
    return None



def get_user_dominant_topic(topics, max_entropy=log2(3), exclude=-1):
    """
    Args:
        topics (list): list of topic labels for a user.
        max_entropy (float): maximum allowed entropy. If user's topic distribution exceeds this, no dominant topic is assigned.
        exclude (int or list): topic(s) to exclude, e.g. -1 for noise.
    
    Returns:
        int or None: dominant topic if above threshold, else None
    """
    if isinstance(exclude, int):
        topics = [t for t in topics if t != exclude]
    elif isinstance(exclude, list):
        topics = [t for t in topics if t not in exclude]

    if not topics:
        return None

    topic_counts = Counter(topics)
    topic_probs = np.array(list(topic_counts.values())) / len(topics)
    
    most_common = topic_counts.most_common(2)
    if len(most_common)>1 and most_common[0][1] == most_common[1][1]:
        return None

    top_topic, _ = most_common[0]
    
    if entropy(topic_probs) < max_entropy:
        return int(top_topic)
    return None

def topic_entropy(topics, exclude = -1):
    """
    Calculate the entropy of a user's topic distribution.
    Args:
        topics (list): list of topic labels for a user.
        exclude (int or list): topic(s) to exclude, e.g. -1 for noise.
    Returns:
        float or None: entropy value if topics are present, else None
    """

    if isinstance(exclude, int):
        topics = [t for t in topics if t != exclude]
    elif isinstance(exclude, list):
        topics = [t for t in topics if t not in exclude]
    if not topics:
        return None
        
    counts = Counter(topics)
    probs = np.array(list(counts.values())) / len(topics)
    return entropy(probs)



# Statistics


def Log2(x):
    return round(log2(x), 2)


def cramers_v(confusion_matrix):
    """
    Calculate Cramer's V for effect size using a confusion matrix, e.g. from pd.crosstab.
    Args:
        confusion_matrix (pd.DataFrame): Confusion matrix as a pandas DataFrame.
    Returns:
        float: Cramer's V statistic.
    """
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    return np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))


def rank_biserial_from_u(u_stat, nx, ny):
    """
    Compute rank-biserial correlation from Mann–Whitney U statistic.
    Formula: r_rb = 1 - (2U) / (n_x * n_y)
    """
    return 1 - (2 * u_stat) / (nx * ny)




def cliffs_delta(x, y):
    """
    Compute Cliff's delta: probability(x > y) - probability(x < y).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    n_x, n_y = len(x), len(y)
    greater = np.sum(x[:, None] > y[None, :])
    less    = np.sum(x[:, None] < y[None, :])
    delta = (greater - less) / (n_x * n_y)
    return delta



def compare_groups(x, y, alternative='two-sided', alpha=0.05):
    """
    Compare two numeric sequences using Mann–Whitney U test,
    Cliff's delta, and rank-biserial correlation.
    Prints results with interpretation.
    """
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()
    
    # Mann–Whitney U
    u_stat, p_value = mannwhitneyu(x, y, alternative=alternative)
    
    # Effect sizes
    delta = cliffs_delta(x, y)
    r_rb = rank_biserial_from_u(u_stat, len(x), len(y))
    
    # Interpretations
    print(f"Sample sizes: n_x = {len(x)}, n_y = {len(y)}")
    print(f"Mann–Whitney U: {u_stat:.3f}, p = {p_value:.5f} (alternative = '{alternative}')")
    
    if p_value < alpha:
        if alternative == 'two-sided':
            direction = "different"
        elif alternative == 'greater':
            direction = "larger" if delta > 0 else "smaller"
        elif alternative == 'less':
            direction = "smaller" if delta < 0 else "larger"
        print(f"→ Statistically significant: x appears {direction} than y (α={alpha})")
    else:
        print(f"→ No statistically significant difference detected at α={alpha}")
    
    # Cliff's delta interpretation
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        magnitude = "\033[34mnegligible\033[0m"
    elif abs_delta < 0.33:
        magnitude = "\033[36msmall\033[0m"
    elif abs_delta < 0.474:
        magnitude = "\033[33mmedium\033[0m"
    else:
        magnitude = "\033[31mlarge\033[0m"
    
    print(f"Cliff's delta: {delta:.3f} ({magnitude} effect)")
    
    # Rank-biserial interpretation (similar scale as correlation)
    print(f"Rank-biserial correlation: {r_rb:.3f} "
          f"(ranges -1 to 1; 0 means no difference)")



# Plotting functions



def plot_entropy_distribution(entropy_series, bins=30, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]):
    """
    Plot the distribution of entropy values from a pandas Series with quantile lines.
    
    Args:
        entropy_series (pd.Series): Entropy values per user.
        bins (int): Number of bins for histogram.
        quantiles (list): List of quantile values to plot (e.g., [0.25, 0.5, 0.75]).
    """
    entropy_series = entropy_series.dropna()
    q_values = entropy_series.quantile(quantiles)

    plt.figure(figsize=(10, 6))
    sns.histplot(entropy_series, bins=bins, kde=True, color='skyblue', edgecolor='black')

    for q, val in zip(quantiles, q_values):
        plt.axvline(x=val, color='red', linestyle='--', linewidth=1.5, label=f'{int(q*100)}th percentile = {val:.2f}')

    # Remove duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title("Distribution of User Topic Entropy with Quantiles")
    plt.xlabel("Entropy")
    plt.ylabel("User Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_effective_topic_count(entropy_series, bins=30, quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]):
    """
    Plot distribution of effective number of topics (2^entropy) with quantile lines.
    
    Args:
        entropy_series (pd.Series): Entropy values per user.
        bins (int): Number of bins in histogram.
        quantiles (list): Quantile values to annotate.
    """
    entropy_series = entropy_series.dropna()
    effective_topics = np.exp2(entropy_series)
    q_values = effective_topics.quantile(quantiles)

    plt.figure(figsize=(10, 6))
    sns.histplot(effective_topics, bins=bins, kde=True, color='skyblue', edgecolor='black')

    for q, val in zip(quantiles, q_values):
        plt.axvline(x=val, color='red', linestyle='--', linewidth=1.5, label=f'{int(q*100)}th percentile = {val:.2f}')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title("Distribution of Effective Number of Topics per User")
    plt.xlabel("Effective Number of Topics (2^entropy)")
    plt.ylabel("User Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()





def main():

    topic_test = [0,0,0,0,0,0,2,2,2,2,2,1,1,1,3,4,5]
    test_topic_counts = Counter(topic_test)
    print(test_topic_counts)



