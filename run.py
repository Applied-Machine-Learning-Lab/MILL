import openai
import argparse
import os
import pyterrier as pt
import pandas as pd
from utils import *
from pyterrier.measures import *
from pandarallel import pandarallel
from tqdm import tqdm
from openai import OpenAI


def main(args):
    # add java11 to os path
    # os.envrion["JAVA_HOME"] = 'your path/java11'
    if not pt.started():
        pt.init()
    
    dataset = pt.datasets.get_dataset(args.dataset)

    if not os.path.exists('./index/{}'.format(args.dataset.replace('/','-'))):
        indexer = pt.IterDictIndexer('./index/{}'.format(args.dataset.replace('/','-')), meta={'docno':200, args.doc_field:4096}, meta_reverse=['docno',args.doc_field])
        indexref = indexer.index(dataset.get_corpus_iter())
        index = pt.IndexFactory.of(indexref)
    else:
        indexref = './index/{}'.format(args.dataset.replace('/', '-'))
        index = pt.IndexFactory.of(indexref)
    
    print('******* queries *******')
    print('query columns: \n{}'.format(dataset.get_topics().columns))
    print('query examples: \n{}'.format(dataset.get_topics().head(5)))
    print('query shape: \n{}'.format(dataset.get_topics().shape))

    print('******* docs *******')
    for line in dataset.get_corpus_iter():
        print(line)
        break
    
    # rewrite
    rewrite_retriever = pt.BatchRetrieve(index, wmodel='BM25', metadata=['docno',args.doc_field], num_results=args.K)

    test_retriever = pt.BatchRetrieve(index, wmodel='BM25', metadata=['docno',args.doc_field])

    eval_metrics = [nDCG@1, nDCG@5, nDCG@10, nDCG@100, nDCG@1000,
                            AP@1, AP@5, AP@10, AP@100, AP@1000,
                            R@1, R@5, R@10, R@100, R@1000,
                            P@1, P@5, P@10, P@100, P@1000,
                            RR@1, RR@5, RR@10, RR@100, RR@1000]

    if args.qe == 'None':
        pipe =  test_retriever
        pipe = pipe.parallel(9)
        results = pt.Experiment(
            [pipe],
            dataset.get_topics(args.query_field).iloc[args.pos_lis[0]:args.pos_lis[1],:] if args.pos_lis else dataset.get_topics(args.query_field),
            dataset.get_qrels(),
            eval_metrics = eval_metrics
        )
    else:
        df = dataset.get_topics('text').copy(deep=True) # if text exist, use it as query; then it will use query automatically
        tokenise = pt.rewrite.tokenise()
        df = tokenise(df).iloc[args.pos_lis[0]:args.pos_lis[1],:] if args.pos_lis else tokenise(df)
        print('******* retrieved results *******')
        retrieve_input = dataset.get_topics(args.query_field).iloc[args.pos_lis[0]:args.pos_lis[1], :] if args.pos_lis else dataset.get_topics(args.query_field)
        retrieved_results = rewrite_retriever(retrieve_input)
        print(retrieved_results.columns)
        print(retrieved_results.head(5))
        print(retrieved_results.shape)
        pandarallel.initialize(progress_bar=True, nb_workers=16)
        df['query'] = df.parallel_apply(lambda row: rewrite(args, args.qe, row, retrieved_results, path_num=args.N, doc_path_num=3, llm_model=args.model_name), axis=1)

        if not os.path.exists('./rewrites/'):
            os.makedirs('./rewrites/')
        df.to_csv('./rewrites/{}-{}-N{}-K{}-n{}-k{}-rewrite.csv'.format(args.dataset.replace('/','-'),args.qe,args.N,args.K,args.n,args.k), index=False)
        results = pt.Experiment(
            [pt.rewrite.tokenise() >> test_retriever.parallel(9)],
            df[['qid','query']],
            dataset.get_qrels(),
            eval_metrics = eval_metrics
        )


    if not os.path.exists('./results/'):
        os.makedirs('./results/')
    print(results)
    results.to_csv('./results/{}_{}_N{}_K{}_n{}_k{}.csv'.format(args.dataset.replace('/','-'), args.qe,args.N,args.K,args.n,args.k))



if __name__ == "__main__":
    client = OpenAI(
        api_key = "",
        base_url = ""
    )
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset','-d', default='irds:beir/scifact/test', type=str, help='refer to https://pyterrier.readthedocs.io/en/latest/datasets.html') 
    # irds:beir/trec-covid irds:beir/msmarco/dev irds:msmarco-passage/trec-dl-2020 irds:beir/webis-touche2020/v2
    # irds:beir/scifact/test irds:beir/nfcorpus/test irds:beir/dbpedia-entity/test irds:beir/scidocs irds:beir/arguana irds:beir/climate-fever
    parser.add_argument('--qe',type=str, default='None',help='None, mill')
    parser.add_argument('--pos_lis', nargs='+', type=int, default=None, help='the index of used queries. usage: --pos_lis 0 25')
    parser.add_argument('--N', type=int, default=3)
    parser.add_argument('-rc','--K', type = int, default =3)
    parser.add_argument('--k', default=3, type=int, help='number of selected PRF documents')
    parser.add_argument('--n', default=3, type = int, help='number of selected LLM generations')
    parser.add_argument('--model_name', type = str, default='gpt-3.5-turbo-instruct', help='text model')

    args = parser.parse_args()

    if args.dataset[:4] != 'irds':
        exit('please choose irds dataset')

    # pre setting
    pd.set_option('display.max_columns', None)
    print('*********** if there are full of THE SERVER IS OVERLOADED OR NOT READY YET, please check your proxy***********')

    args.query_field = 'text'
    
    if args.dataset == 'irds:msmarco-document/trec-dl-2020':
        args.doc_field = 'body'
    else:
        args.doc_field = 'text'

    main(args)