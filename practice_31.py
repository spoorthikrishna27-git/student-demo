# ==========================================
# Practice 31: LDA Model Evaluation
# Using Coherence Score
# ==========================================

from multiprocessing import freeze_support

import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel


def main():

    # ------------------------------------------
    # Step 1: Load Dataset
    # ------------------------------------------

    categories = [
        'alt.atheism',
        'soc.religion.christian',
        'comp.graphics',
        'sci.med'
    ]

    newsgroups = fetch_20newsgroups(
        subset='train',
        categories=categories,
        shuffle=True,
        random_state=42,
        remove=('headers', 'footers', 'quotes')
    )

    print("Dataset Loaded Successfully")
    print("Number of Documents:", len(newsgroups.data))

    # ------------------------------------------
    # Step 2: Create Bag of Words
    # ------------------------------------------

    count_vectorizer = CountVectorizer(
        stop_words='english',
        max_df=0.95,
        min_df=2
    )

    count_matrix = count_vectorizer.fit_transform(newsgroups.data)

    print("\nShape of Count Matrix:")
    print(count_matrix.shape)

    # ------------------------------------------
    # Step 3: Prepare Data
    # ------------------------------------------

    texts = [doc.lower().split() for doc in newsgroups.data]

    dictionary = Dictionary(texts)

    feature_names = count_vectorizer.get_feature_names_out()

    # ------------------------------------------
    # Step 4: Train LDA Models
    # ------------------------------------------

    topic_numbers = [2, 3, 5, 7, 10]

    coherence_scores = []

    print("\nTraining LDA Models\n")

    for num_topics in topic_numbers:

        lda_model = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42
        )

        lda_model.fit(count_matrix)

        topics = []

        for topic in lda_model.components_:

            top_indices = topic.argsort()[-10:][::-1]

            topic_words = [
                feature_names[i]
                for i in top_indices
            ]

            topics.append(topic_words)

        coherence_model = CoherenceModel(
            topics=topics,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v',
            processes=1
        )

        score = coherence_model.get_coherence()

        coherence_scores.append(score)

        print(f"{num_topics} Topics --> Coherence Score = {score:.4f}")

    # ------------------------------------------
    # Step 5: Compare Scores
    # ------------------------------------------

    print("\nComparison of Coherence Scores\n")

    for topic, score in zip(topic_numbers, coherence_scores):

        print(f"{topic} Topics : {score:.4f}")

    # ------------------------------------------
    # Step 6: Best Number of Topics
    # ------------------------------------------

    best_score = max(coherence_scores)

    best_topics = topic_numbers[
        coherence_scores.index(best_score)
    ]

    print("\nOptimal Number of Topics:", best_topics)

    print("Highest Coherence Score:",
          round(best_score, 4))

    # ------------------------------------------
    # Step 7: Plot Graph
    # ------------------------------------------

    plt.figure(figsize=(8,5))

    plt.plot(
        topic_numbers,
        coherence_scores,
        marker='o',
        linewidth=2
    )

    plt.title("Number of Topics vs Coherence Score")
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")

    plt.grid(True)

    plt.show()

    # ------------------------------------------
    # Step 8: Interpretation
    # ------------------------------------------

    print("\nInterpretation\n")

    print("1. Higher coherence score indicates better topic quality.")

    print("2. Topics with similar words have higher coherence.")

    print("3. The model with the highest coherence score is considered the best model.")

    print("4. Too few topics combine different concepts into one topic.")

    print("5. Too many topics split meaningful topics into smaller groups.")

    print("6. Coherence score helps us choose the optimal number of topics in topic modeling.")


if __name__ == "__main__":

    freeze_support()

    main()