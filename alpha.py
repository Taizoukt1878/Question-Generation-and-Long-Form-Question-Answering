#import pandas as pd
import streamlit as st
from pipelines import pipeline
from PyPDF2 import PdfReader
#haystack
from haystack.document_stores import PineconeDocumentStore
from haystack import Document
from haystack.nodes import Seq2SeqGenerator
from haystack.nodes import EmbeddingRetriever
from haystack.pipelines import GenerativeQAPipeline



st.set_page_config(layout="wide")


#Functions

def connect_to_doc_store():
	document_store = PineconeDocumentStore(
    api_key='Here you should put your pincone api_key',
    index='here you should put your index name',
    similarity="cosine",
    embedding_dim=768
    )
	return document_store

def get_retriever(document_store):

	retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="flax-sentence-embeddings/all_datasets_v3_mpnet-base",
    model_format="sentence_transformers"
	)
	return retriever

def get_generator():
	generator = Seq2SeqGenerator(model_name_or_path="vblagoje/bart_lfqa")
	return generator

def get_pipeline(generator, retriever):
	pipe = GenerativeQAPipeline(generator, retriever)
	return pipe

def update_doc_store(doc: str, article_title : str = "None", section_title : str = "None" ):
	docs = []
	doc = Document(
        	content= doc,
        	meta={
            	"article_title": article_title,
            	'section_title': section_title
        	}
    	)
	docs.append(doc)
	embeds = retriever.embed_documents(docs)
	doc.embedding = embeds[0]
	document_store.write_documents(docs)

def text_from_pdf_qg(path: str):
	nlp = pipeline("e2e-qg")
	reader = PdfReader(path)
	pages = reader.pages
	questions = []
	all_doc = str()
	for page in pages:
		doc = page.extract_text()
	#	all_doc = all_doc + doc + " "
		quest = nlp(doc)
		for item in quest:
			questions.append(item)
	#update_doc_store(doc)
	return questions


def text_from_pdf(path: str) -> str:
	reader = PdfReader(path)
	pages = reader.pages
	doc = str()
	for page in pages:
		doc = doc + page.extract_text() + " "
	update_doc_store(doc)

def get_answer(question):
	document_store = connect_to_doc_store()
	retriever = get_retriever(document_store)
	generator = get_generator()
	pipe = get_pipeline(generator, retriever)
	result = pipe.run(
	query=str(question),
	params={
        "Retriever": {"top_k": 3},
        "Generator": {"top_k": 1}
    	}
	)
	answer_dic = result["answers"][0].to_dict()
	return answer_dic






# Page containers

header = st.container()

section1 = st.container()

section2 = st.container()

section3 = st.container()

section4 = st.container()






# Header section

with header:
	st.title("Question Generation, and Long Form Question Answering")


with section1:

	#col1, col2 = st.columns(2)

	# Reading pdf files
	#with col1 :
	st.title("PDF")
	file = st.file_uploader("Upload a pdf file")
		#text_from_pdf(file)
	# Writing a question
	#with col2:
	st.title("input question")
	question_input = st.text_input("Ask a question")

# Generating questions

with section2:
	if(file):
		questions = text_from_pdf_qg(file)
		st.title("Generated question")
		question_gen = st.radio("Generated questions", questions)
		st.write(question_gen)


with section3:
	if st.button("Submit generated question"):
		st.write("question: ", question_gen)
		answer_dic = get_answer(question_gen)
		st.subheader("Answer")
		st.write(str(answer_dic["answer"]))

with section4:
		if st.button("Submit question"):
			st.write("question: ", question_input)
			answer_dic = get_answer(question_input)
			st.subheader("Answer")
			st.write(str(answer_dic["answer"]))


