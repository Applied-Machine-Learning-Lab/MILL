import openai
import pyterrier as pt
from retrying import retry
import json
import time
import random
import numpy as np
from openai import OpenAI
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

@retry(stop_max_attempt_number=10, wait_fixed=2000)
def get_completions_response(messages, temperature=0.7, n=1, max_tokens=500, model='gpt-3.5-turbo-instruct'):
    if isinstance(messages, list):
        input_str = ''
        for m in messages:
            input_str += m['content'] + ' '
    else: 
        input_str = messages
    client = OpenAI(
        api_key = "",
        base_url = ""
    )
    response = client.completions.create(
        model = model,
        prompt = input_str,
        max_tokens= max_tokens,
        temperature = temperature,
        n = n,
        stop = None
    )
    answer = []
    for ans in response.choices:
        answer.append(ans.text)
    return answer


@retry(stop_max_attempt_number=3, wait_fixed=2000)
def get_embedding(messages, model='text-embedding-ada-002'):
    client = OpenAI(
        api_key = "",
        base_url = ""
    )
    embedding = client.embeddings.create(
        input=messages,
        model=model
    )
    return embedding.data[0].embedding

@retry(stop_max_attempt_number=3, wait_fixed=2000)
def rewrite(args, method_name, x, retrieved_results, path_num, doc_path_num, llm_model):
    query = x.query
    if llm_model in ['text-davinci-003','gpt-3.5-turbo-instruct']:
        rewrite_function = get_completions_response
    try:
        if method_name == 'mill':
            rewritten_list =  mill_rewrite(args, query, retrieved_results, path_num, doc_path_num, llm_model, args.k, args.n, rewrite_function)
        rewritten_query = ''.join(rewritten_list)
        return rewritten_query
    except Exception as e:
        print(e)
        rewritten_query = query
        return query

def mill_rewrite(args, query, retrieved_results, path_num, doc_path_num, llm_model, k, n, rewrite_function):
    rewritten_list = []
    doc_lis = retrieved_results[retrieved_results['query']==query][args.doc_field].tolist()
    doc_lis = np.array([x[:1024] for x in doc_lis])
    # for path_idx in range(path_num):
    prompt = []
    prompt.append({"role": "system", "content": "What subquestion should be searched to answer the following query: "+query +'. I will generate the subquestions and write passages to answer these generated questions:'})
    response = rewrite_function(prompt, model=llm_model, n=path_num)
    for path_idx in range(path_num):
        rewritten_list.append(response[path_idx])
    rewritten_list = np.array(rewritten_list)
    # get embeddings for rewritten_list
    rewritten_embedding_list = []
    for rewrite in rewritten_list:
        embedding = get_embedding(rewrite)
        rewritten_embedding_list.append(embedding)
    rewritten_embedding_list = np.array(rewritten_embedding_list)
    # get embeddings for the retrieved docs
    retrieved_embedding_list = []
    for doc in doc_lis:
        embedding = get_embedding(doc)
        retrieved_embedding_list.append(embedding)
    retrieved_embedding_list = np.array(retrieved_embedding_list)
    # cal similarities between rewritten embeddings between selected docs
    similarity_lis = []
    for rewritten_embedding in rewritten_embedding_list:
        rew_similarity = 0
        for retrieved_embedding in retrieved_embedding_list:
            similarity = cosine_similarity(rewritten_embedding.reshape(1,-1), retrieved_embedding.reshape(1,-1))
            rew_similarity += similarity[0][0]
        similarity_lis.append(rew_similarity)
    similarity_lis = np.array(similarity_lis)
    selected_rewritten = rewritten_list[np.argsort(similarity_lis)[-n:]]
    # selected_rewritten_embedding = rewritten_embedding_list[np.argsort(similarity_lis)[-3:]]
    # cal similarities between retrieved and rewritten embeddings
    similarity_lis = []
    for retrieved_embedding in retrieved_embedding_list:
        doc_similarity = 0
        for rewritten_embedding in rewritten_embedding_list:
            similarity = cosine_similarity(retrieved_embedding.reshape(1,-1), rewritten_embedding.reshape(1,-1))
            doc_similarity += similarity[0][0]
        similarity_lis.append(doc_similarity)
    similarity_lis = np.array(similarity_lis)

    selected_doc = doc_lis[np.argsort(similarity_lis)[-k:]]
    
    res = [query*5]
    res.append(' '.join(selected_doc[::-1]))
    res.append(' '.join(selected_rewritten[::-1]))
    return res
