# %% [markdown]
# # Evaluation
#
# The plans is to run the evaluations using our newly created dataset while varying
# several factors:
#   * basic (vector) vs. advanced (vector + bm25) search
#   * contextualized vs. non-contextualized chunks
#   * number of records retrieved (k=5, k=10, k=20)
#
# We calculate these metrics to evaluate retrieval:
#   * precision
#   * recall
#   * F1 score
#   * mean reciprocal rank (MRR)
#
# And for evaluating end-to-end performance:
#   * end-to-end accuracy (LLM-as-a-judge)
#

# %%
import json
from pathlib import Path
from tqdm import tqdm
from textwrap import dedent
from typing import Callable, Tuple

from elasticsearch import Elasticsearch
from es import get_es_client
from google import genai
from google.genai.types import GenerateContentConfig
from goog import get_google_client, MODEL
from pandas import DataFrame
from pydantic import BaseModel

from rag import end_to_end_advanced, end_to_end_basic, retrieve_advanced, retrieve_basic
from utils import retry, write_to_jsonl

# %%
def evaluate_retrieval(eval_data: list[dict], retrieval_fn: Callable, client: Elasticsearch, k: int, contextual: bool, similarity_threshold: float | None = None, start_message: str = '') -> dict[str, float]:
    '''
    Evaluate retrieval by comparing retrieved chunks to the golden chunk
    for each query in the eval dataset
    '''

    print(start_message)

    precisions = []
    recalls = []
    mrrs = []

    for item in tqdm(eval_data, desc='evaluating retrieval'):
        query = item['query']
        golden_chunk_id = item['chunk_id']

        # retrieve chunks
        retrieved_chunks = retrieval_fn(
            client,
            query,
            k=k,
            contextual=contextual,
            similarity_threshold=similarity_threshold
        )
        retrieved_chunk_ids = [chunk['_source']['chunk_id'] for chunk in retrieved_chunks]

        # calculate metrics
        # 1 if golden chunk in the retrieved chunks, 0 otherwise
        true_positives = 1 if golden_chunk_id in retrieved_chunk_ids else 0
        # precision: 1 if golden chunk is the only chunk retrieved, lower otherwise
        precision = true_positives / len(retrieved_chunk_ids) if retrieved_chunk_ids else 0
        # recall: 1 if golden chunk in the retrieved chunks, 0 otherwise
        recall = true_positives / 1 # len(golden_chunk_ids), except we only have one golden chunk per query
        # mean reciprocal rank: 1 / (position of golden chunk in the retrieved chunks), 0 if not present
        mrr = 1 / (retrieved_chunk_ids.index(golden_chunk_id) + 1) if golden_chunk_id in retrieved_chunk_ids else 0

        precisions.append(precision)
        recalls.append(recall)
        mrrs.append(mrr)

    # calculate averages
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_mrr = sum(mrrs) / len(mrrs)
    f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0

    result = {
        'precision': avg_precision,
        'recall': avg_recall,
        'mrr': avg_mrr,
        'f1': f1,
        'total_queries': len(eval_data)
    }
    print(result)
    return result


class EndToEndEvaluation(BaseModel):
    is_correct: bool
    explanation: str


def evaluate_end_to_end(eval_data: list[dict], generate_answer_fn: Callable, client_es: Elasticsearch, client_goog: genai.Client, k: int, contextual: bool, similarity_threshold: float | None = None, start_message: str = '') -> Tuple[float, list[dict]]:

    correct_answers = 0
    results = []
    total_questions = len(eval_data)

    print(start_message)

    for i, item in enumerate(tqdm(eval_data, desc='evaluating end-to-end')):
        query = item['query']
        correct_answer = item['answer']

        generated_answer = retry(generate_answer_fn, (client_es, client_goog, query, k, contextual, similarity_threshold))

        eval_prompt = dedent(f'''
            You are an AI assistant tasked with evaluating the correctness of answers to questions about different Texas Instruments calculators.

            Question: {query}

            Correct Answer: {correct_answer}

            Generated Answer: {generated_answer}

            Is the Generated Answer correct based on the Correct Answer? You should pay attention to the substance of the answer, and ignore minute details that may differ.

            Small differences or changes in wording don't matter. If the generated answer and correct answer are saying essentially the same thing then that generated answer should be marked correct.

            However, if there is any critical piece of information which is missing from the generated answer in comparison to the correct answer, then we should mark this as incorrect.

            Finally, if there are any direct contradictions between the correct answer and generated answer, we should deem the generated answer to be incorrect.

            Using the response schema provided, give your judgement on the correctness of the generated answer, and provide an explanation for your decision.
        ''')

        def _generate_eval(client: genai.Client, prompt: str) -> EndToEndEvaluation:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=GenerateContentConfig(
                    temperature=0.0,  # minimum creativity
                    response_mime_type='application/json',
                    response_schema=EndToEndEvaluation,
                ),
            )
            if response.parsed:
                return response.parsed
            print(f'Error parsing evaluation response for question {i}, assuming incorrect.')
            return EndToEndEvaluation(is_correct=False, explanation=response.text)

        evaluation = retry(_generate_eval, (client_goog, eval_prompt))

        if evaluation.is_correct:
            correct_answers += 1
        results.append(evaluation.model_dump())

        print(f'Question {i + 1}/{total_questions}: {query}')
        print(f'Correct Answer: {correct_answer}')
        print(f'Generated Answer: {generated_answer}')
        print(f'Correct: {evaluation.is_correct}')
        print('-'*88)

        if (i + 1) % 10 == 0:
            current_accuracy = correct_answers / (i + 1)
            print(f'Processed {i + 1}/{total_questions} questions. Current Accuracy: {current_accuracy:.4f}')

    accuracy = correct_answers / total_questions
    return accuracy, results

# %%
project_root = Path(__file__).parent.parent
eval_dataset = project_root / 'data' / 'tech-manual-rag.eval.jsonl'

with open(eval_dataset, 'r') as f:
    eval_data = [json.loads(line) for line in f]

client_es = get_es_client()
client_goog = get_google_client()

# %%
# ######################### retrieval evaluation #########################
df = DataFrame({'desc': [], 'eval': []})

# basic retrieval, non-contextualized
desc = 'basic retrieval, non-contextualized, k=20'
result = evaluate_retrieval(eval_data, retrieve_basic, client_es, k=20, contextual=False, start_message=desc)
df.loc[df.shape[0]] = [desc, result]

desc = 'basic retrieval, non-contextualized, k=10'
result = evaluate_retrieval(eval_data, retrieve_basic, client_es, k=10, contextual=False, start_message=desc)
df.loc[df.shape[0]] = [desc, result]

desc = 'basic retrieval, non-contextualized, k=5'
result = evaluate_retrieval(eval_data, retrieve_basic, client_es, k=5, contextual=False, start_message=desc)
df.loc[df.shape[0]] = [desc, result]

# basic retrieval, contextualized
desc = 'basic retrieval, contextualized, k=20'
result = evaluate_retrieval(eval_data, retrieve_basic, client_es, k=20, contextual=True, start_message=desc)
df.loc[df.shape[0]] = [desc, result]

desc = 'basic retrieval, contextualized, k=10'
result = evaluate_retrieval(eval_data, retrieve_basic, client_es, k=10, contextual=True, start_message=desc)
df.loc[df.shape[0]] = [desc, result]

desc = 'basic retrieval, contextualized, k=5'
result = evaluate_retrieval(eval_data, retrieve_basic, client_es, k=5, contextual=True, start_message=desc)
df.loc[df.shape[0]] = [desc, result]

write_to_jsonl(df.to_dict('records'), project_root / 'data' / 'eval-retrieval-basic.jsonl')

df = DataFrame({'desc': [], 'eval': []})

# advanced retrieval, non-contextualized
desc = 'advanced retrieval, non-contextualized, k=20'
result = evaluate_retrieval(eval_data, retrieve_advanced, client_es, k=20, contextual=False, start_message=desc)
df.loc[df.shape[0]] = [desc, result]

desc = 'advanced retrieval, non-contextualized, k=10'
result = evaluate_retrieval(eval_data, retrieve_advanced, client_es, k=10, contextual=False, start_message=desc)
df.loc[df.shape[0]] = [desc, result]

desc = 'advanced retrieval, non-contextualized, k=5'
result = evaluate_retrieval(eval_data, retrieve_advanced, client_es, k=5, contextual=False, start_message=desc)
df.loc[df.shape[0]] = [desc, result]

# advanced retrieval, contextualized
desc = 'advanced retrieval, contextualized, k=20'
result = evaluate_retrieval(eval_data, retrieve_advanced, client_es, k=20, contextual=True, start_message=desc)
df.loc[df.shape[0]] = [desc, result]

desc = 'advanced retrieval, contextualized, k=10'
result = evaluate_retrieval(eval_data, retrieve_advanced, client_es, k=10, contextual=True, start_message=desc)
df.loc[df.shape[0]] = [desc, result]

desc = 'advanced retrieval, contextualized, k=5'
result = evaluate_retrieval(eval_data, retrieve_advanced, client_es, k=5, contextual=True, start_message=desc)
df.loc[df.shape[0]] = [desc, result]

write_to_jsonl(df.to_dict('records'), project_root / 'data' / 'eval-retrieval-advanced.jsonl')

# %%
######################### end-to-end evaluation #########################
df = DataFrame({'desc': [], 'accuracy': []})

# end-to-end, basic retrieval, non-contextualized
desc = 'end-to-end, basic retrieval, non-contextualized, k=20'
accuracy, results = evaluate_end_to_end(eval_data, end_to_end_basic, client_es, client_goog, k=20, contextual=False, start_message=desc)
df.loc[df.shape[0]] = [desc, accuracy]
write_to_jsonl(results, Path(f'./data/eval-{desc.replace(',', '').replace(" ", "-")}.jsonl'))

desc = 'end-to-end, basic retrieval, non-contextualized, k=10'
accuracy, results = evaluate_end_to_end(eval_data, end_to_end_basic, client_es, client_goog, k=10, contextual=False, start_message=desc)
df.loc[df.shape[0]] = [desc, accuracy]
write_to_jsonl(results, Path(f'./data/eval-{desc.replace(',', '').replace(" ", "-")}.jsonl'))

desc = 'end-to-end, basic retrieval, non-contextualized, k=5'
accuracy, results = evaluate_end_to_end(eval_data, end_to_end_basic, client_es, client_goog, k=5, contextual=False, start_message=desc)
df.loc[df.shape[0]] = [desc, accuracy]
write_to_jsonl(results, Path(f'./data/eval-{desc.replace(',', '').replace(" ", "-")}.jsonl'))

# end-to-end, basic retrieval, contextualized
desc = 'end-to-end, basic retrieval, contextualized, k=20'
accuracy, results = evaluate_end_to_end(eval_data, end_to_end_basic, client_es, client_goog, k=20, contextual=True, start_message=desc)
df.loc[df.shape[0]] = [desc, accuracy]
write_to_jsonl(results, Path(f'./data/eval-{desc.replace(',', '').replace(" ", "-")}.jsonl'))

desc = 'end-to-end, basic retrieval, contextualized, k=10'
accuracy, results = evaluate_end_to_end(eval_data, end_to_end_basic, client_es, client_goog, k=10, contextual=True, start_message=desc)
df.loc[df.shape[0]] = [desc, accuracy]
write_to_jsonl(results, Path(f'./data/eval-{desc.replace(',', '').replace(" ", "-")}.jsonl'))

desc = 'end-to-end, basic retrieval, contextualized, k=5'
accuracy, results = evaluate_end_to_end(eval_data, end_to_end_basic, client_es, client_goog, k=5, contextual=True, start_message=desc)
df.loc[df.shape[0]] = [desc, accuracy]
write_to_jsonl(results, Path(f'./data/eval-{desc.replace(',', '').replace(" ", "-")}.jsonl'))

write_to_jsonl(df.to_dict('records'), project_root / 'data' / 'eval-end-to-end-basic.jsonl')

df = DataFrame({'desc': [], 'accuracy': []})

# end-to-end, advanced retrieval, non-contextualized
desc = 'end-to-end, advanced retrieval, non-contextualized, k=20'
accuracy, results = evaluate_end_to_end(eval_data, end_to_end_advanced, client_es, client_goog, k=20, contextual=False, start_message=desc)
df.loc[df.shape[0]] = [desc, accuracy]
write_to_jsonl(results, Path(f'./data/eval-{desc.replace(',', '').replace(" ", "-")}.jsonl'))

desc = 'end-to-end, advanced retrieval, non-contextualized, k=10'
accuracy, results = evaluate_end_to_end(eval_data, end_to_end_advanced, client_es, client_goog, k=10, contextual=False, start_message=desc)
df.loc[df.shape[0]] = [desc, accuracy]
write_to_jsonl(results, Path(f'./data/eval-{desc.replace(',', '').replace(" ", "-")}.jsonl'))

desc = 'end-to-end, advanced retrieval, non-contextualized, k=5'
accuracy, results = evaluate_end_to_end(eval_data, end_to_end_advanced, client_es, client_goog, k=5, contextual=False, start_message=desc)
df.loc[df.shape[0]] = [desc, accuracy]
write_to_jsonl(results, Path(f'./data/eval-{desc.replace(',', '').replace(" ", "-")}.jsonl'))

# end-to-end, advanced retrieval, contextualized
desc = 'end-to-end, advanced retrieval, contextualized, k=20'
accuracy, results = evaluate_end_to_end(eval_data, end_to_end_advanced, client_es, client_goog, k=20, contextual=True, start_message=desc)
df.loc[df.shape[0]] = [desc, accuracy]
write_to_jsonl(results, Path(f'./data/eval-{desc.replace(',', '').replace(" ", "-")}.jsonl'))

desc = 'end-to-end, advanced retrieval, contextualized, k=10'
accuracy, results = evaluate_end_to_end(eval_data, end_to_end_advanced, client_es, client_goog, k=10, contextual=True, start_message=desc)
df.loc[df.shape[0]] = [desc, accuracy]
write_to_jsonl(results, Path(f'./data/eval-{desc.replace(',', '').replace(" ", "-")}.jsonl'))

desc = 'end-to-end, advanced retrieval, contextualized, k=5'
accuracy, results = evaluate_end_to_end(eval_data, end_to_end_advanced, client_es, client_goog, k=5, contextual=True, start_message=desc)
df.loc[df.shape[0]] = [desc, accuracy]
write_to_jsonl(results, Path(f'./data/eval-{desc.replace(',', '').replace(" ", "-")}.jsonl'))

write_to_jsonl(df.to_dict('records'), project_root / 'data' / 'eval-end-to-end-advanced.jsonl')

