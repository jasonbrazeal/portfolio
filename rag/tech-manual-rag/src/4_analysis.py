# %% [markdown]
# # Analysis
#
# ## More about the metrics
#
# ### Precision
# Measures the proportion of retrieved chunks that were relevant to the query. Precision is higher when most retrieved chunks are relevant, and lower when many irrelevant chunks are present. In our system, we only have one golden chunk per query, and we retrieve a fixed set of chunks per query (k=5, 10, 20), which will definitely affect precision. For example, if our system works perfectly and retrieves the golden chunk first for a query at k=5, precision would be calculated as 1/5 (0.2). We can use average precision as a metric to compare the different retrieval approaches, as long as we keep this in mind.
#
# ### Recall
# Measures how many golden chunks we were able to retrieve. Higher is better because it means fewer golden chunks were missed. Recall is especially important for RAG systems since essential context missing from the prompt can negatively affect the LLM performance. In our case, recall for single query is 1 if the single golden chunk is present in the retrieved chunks and 0 otherwise.
#
# ### F1 score
# Harmonic mean of precision and recall, providing a balanced measure of overall retrieval quality. While recall is more critical for RAG systems, the F1 score gives us a useful single metric to evaluate retrieval across different configurations.
#
# ### Mean Reciprocal Rank (MRR)
# Measures how well the system ranks relevant chunks. It considers the rank of the golden chunk for each query and can range from 0 to 1, with 1 meaning the golden chunk is always first.
#
# ### End-to-end Accuracy
# Measures the overall system accuracy: how often the LLM, when given retrieved context, produces an answer judged correct by an LLM-as-a-judge. This metric reflects both retrieval and generation quality.


# %%
import json
from pathlib import Path

from pandas import DataFrame

from utils import create_chart

# %%
project_root = Path(__file__).parent.parent
data_dir = project_root / 'data'
pdf_dir = data_dir / 'texas_instruments_manuals'
dataset_orig = data_dir / 'tech-manual-rag.jsonl'
dataset_contextualized = data_dir / 'tech-manual-rag.contextualized.jsonl'
dataset_embedded = data_dir / 'tech-manual-rag.contextualized.embedded.jsonl'
img_dir = project_root / 'img'
img_dir.mkdir(exist_ok=True)

# %%
retrieval_rows = []
retrieval_files = sorted(data_dir.glob('eval-retrieval-*.jsonl'))
for filepath in retrieval_files:
    with filepath.open() as f:
        for line in f:
            d = json.loads(line)
            desc = d['desc']
            evaluation = d['eval']
            parts = desc.split(',') # "basic retrieval, contextualized, k=10"
            retrieval_type = parts[0].split()[0].strip()
            contextualized = parts[1].strip() == 'contextualized'
            k = int(parts[-1].split('=')[-1])
            retrieval_rows.append({
                'retrieval_type': retrieval_type,
                'contextualized': contextualized,
                'k': k,
                **evaluation
            })
df_retrieval = DataFrame(retrieval_rows)

# %% [markdown]
# ### Retrieval eval data
# ```text
#    retrieval_type  contextualized   k  precision    recall       mrr        f1  total_queries
# 0        advanced           False  20   0.035000  0.700000  0.344744  0.066667            180
# 1        advanced           False  10   0.059444  0.594444  0.335273  0.108081            180
# 2        advanced           False   5   0.091111  0.455556  0.310000  0.151852            180
# 3        advanced            True  20   0.046389  0.927778  0.554566  0.088360            180
# 4        advanced            True  10   0.085556  0.855556  0.546102  0.155556            180
# 5        advanced            True   5   0.152222  0.761111  0.530556  0.253704            180
# 6           basic           False  20   0.028889  0.577778  0.311471  0.055026            180
# 7           basic           False  10   0.050556  0.505556  0.306609  0.091919            180
# 8           basic           False   5   0.076667  0.383333  0.290000  0.127778            180
# 9           basic            True  20   0.041667  0.833333  0.520340  0.079365            180
# 10          basic            True  10   0.075556  0.755556  0.514381  0.137374            180
# 11          basic            True   5   0.136667  0.683333  0.504907  0.227778            180
# ```

# %% [markdown]
# ## Interpreting retrieval results
#
# ### Precision-Recall trade-off
# There is often a trade-off between precision and recall, and we see that in our system with different values of k. Higher values of k produce higher recall; the relevant chunk is more likely to be retrieved if you retrieve more chunks. But precision suffers with higher values of k because only 1 of the retrieved chunks can be the golden chunk. In this RAG system, our main goal is to maximize recall. This ensures that the LLM has access to as much relevant information as possible, even if some irrelevant chunks are included. High recall is very important for technical manuals, where missing a key instruction or detail can lead to poor downstream results.
#
# ### Basic vs. advanced retrieval
# The basic vector search did not perform as well as the advanced hybrid search, which combines vector and lexical (BM25) search. The hybrid search consistently achieves higher recall and MRR than the basic vector search approach. This means hybrid search is more effective at surfacing relevant information from the manuals.
#
# ### Contextualized vs. non-contextualized chunks
# Contextualized chunks performed better than non-contextualized chunks. For both retrieval types, using contextualized chunks (where each chunk is embedded with additional context about its place in the document) leads to higher recall, precision, and MRR. Situating chunks within their broader context helps the system retrieve more relevant results.
#
# ### Number of chunks retrieved (k=5, k=10, k=20)
# Increasing k leads to higher recall and F1 scores, though precision drops slightly. As mentioned above, retrieving more chunks increases the chance of including the relevant one, but also brings in more irrelevant chunks. Since we prioritize recall, higher k values are preferred in this setting.
#

# %%
end_to_end_rows = []
end_to_end_files = [data_dir / 'eval-end-to-end-advanced.jsonl', data_dir / 'eval-end-to-end-basic.jsonl']
for filepath in end_to_end_files:
    with filepath.open() as f:
        for line in f:
            e = json.loads(line)
            if not e:
                continue
            desc = e['desc']
            accuracy = e['accuracy']
            parts = desc.split(', ') # "end-to-end, advanced retrieval, non-contextualized, k=10"
            retrieval_type = parts[1].split()[0].strip()
            contextualized = parts[2].strip() == 'contextualized'
            k = int(parts[-1].split('=')[-1])
            end_to_end_rows.append({
                'retrieval_type': retrieval_type,
                'contextualized': contextualized,
                'k': k,
                'accuracy': accuracy,
            })

df_end_to_end = DataFrame(end_to_end_rows).sort_values('accuracy', ascending=False)

# %% [markdown]
# ### End-to-end eval data
# ```text
#    retrieval_type  contextualized   k  accuracy
# 4        advanced            True  10  0.961111
# 3        advanced            True  20  0.955556
# 5        advanced            True   5  0.944444
# 10          basic            True  10  0.938889
# 9           basic            True  20  0.933333
# 11          basic            True   5  0.916667
# 0        advanced           False  20  0.900000
# 1        advanced           False  10  0.872222
# 6           basic           False  20  0.866667
# 7           basic           False  10  0.855556
# 2        advanced           False   5  0.822222
# 8           basic           False   5  0.805556
# ```

# %% [markdown]
# ## Interpreting end-to-end results
#
# Contextualized chunks consistently outperform non-contextualized chunks for both retrieval types (basic and advanced). Situating chunks within their document context helps the LLM generate more accurate answers.
#
# Advanced retrieval is always better than basic retrieval. For every value of k and contextualization, advanced retrieval yields higher accuracy. This matches the retrieval metrics and confirms that combining vector and BM25 search surfaces more relevant information for the LLM.
#
# k=10 is the sweet spot for our dataset. For both advanced and basic retrieval, k=10 gives the highest accuracy with contextualized chunks (0.961 and 0.939, respectively). Increasing k to 20 slightly decreases accuracy, likely because too much context introduces noise or distracts the LLM.
#
# Accuracy is very high for the best configurations. The top result (advanced, contextualized, k=10) achieves over 96% accuracy, indicating that the system is highly effective at answering questions from the manuals when using the best retrieval strategy.

# %% [markdown]
#
# ### Comparing performance: Basic RAG vs. Advanced RAG with contextual embeddings

# %%
create_chart(df_retrieval, df_end_to_end, img_dir / 'evaluation.png')

# %% [markdown]
#
# ### Performance improvements
#
# Advanced RAG with contextual embeddings improved the golden chunk retrieval rate (recall) by 60% (0.58 --> 0.93) and the end-to-end accuracy by 10% (0.87 --> 0.96)
#
# ![Basic RAG vs. Advanced RAG with contextual embeddings](../img/evaluation.png)


# %% [markdown]
# ## Conclusion
#
# The results highlight the importance of search retrieval quality and chunk context in a RAG system. Improvements in either can have a significant impact on downstream LLM performance. For technical manuals, maximizing recall (as seen in retrieval metrics) is important, but too much context can hurt end-to-end accuracy. There is a balance between providing enough relevant information and avoiding information overload. These findings are consistent with Anthropic's results and highlight the value of contextual retrieval for technical documentation.
