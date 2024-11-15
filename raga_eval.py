import os
import pandas as pd
from datasets import load_dataset, Dataset


from ragas import EvaluationDataset, SingleTurnSample
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, LLMContextPrecisionWithReference
from ragas import evaluate


from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings


from dotenv import load_dotenv

load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_KEY")

df = pd.read_json("dataset.json")

dataset = Dataset.from_pandas(df)

questions = df['question'].to_list()
answers = df['answer'].to_list()
contexts = df['contexts'].to_list()
ground_truths = df['ground_truths'].to_list()

samples = []

for user_input, retrieved_contexts, response, reference in zip(questions, contexts, answers, ground_truths):

    data = SingleTurnSample(user_input=user_input, retrieved_contexts=retrieved_contexts, response=response, reference=reference[0])

    samples.append(data)

eval_dataset = EvaluationDataset(samples=samples)

print(eval_dataset)



evaluator_llm = LangchainLLMWrapper(ChatOpenAI(temperature = 0.0, model="gpt-4o-mini",  api_key=OPENAI_API_KEY))
evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=OPENAI_API_KEY))

metrics = [
    LLMContextRecall(llm=evaluator_llm), 
    LLMContextPrecisionWithReference(llm=evaluator_llm),
    FactualCorrectness(llm=evaluator_llm)
]
results = evaluate(dataset=eval_dataset, metrics=metrics)

df = results.to_pandas()

print(df.head())
df.to_excel("report.xlsx")
