# Long Form Question Aswering & Question Generation 
# By Anouar Taizoukt

- [Long Form Question Aswering & Question Generation using 🤗transformers](#Long-Form-Question-Aswering-&-Question-Generation)
  - [Project Details](#project-details)
  - [Question Generation](#question-generation)
  - [Question Answering](#question-answering)
  - [Vector Database](#vector-Data-Base)
  - [project workflow](#project-workflow)
  - [files description](#files-descreption)
  - [How to excute the app](How-to-excute-the-app)
  - [pics from the app](pics-from-the-app)




## Project details:

Question generation is the task of automatically generating questions from a text document, while qustion answering/long form question answering is the task of automatically answering questions based on a given document or on the knowledge of a question answering pre-trained model.



while there is many methodes we can adopte to performe both of this tasks, the ones we will be using are :

## Question Generation

For the question generation our strategy will be the end-to-end question generation, in which we will use a pre-trained t5 model to generate the questions directly by providing a context. the output of this model is a paragraph that contains all the generated questions separated by <spe>.

## Question Answering

For the question answering, and in order to guarrante a well explained answers our strategy will be as follows:

- Firstly we will keep the documents provided by each user in a vector Data Base in order to keep incresing the knowledge of our app, the stored documents will be for sure helpful to provide a well explained answers to the users.

- secondly, and in order to guarantee that our LFQA model won't use all the stored documents while answering a question because that would lead to two big issues, the first one is that the answer might be inaccurate and the seconde one is that the processing will take too much time because the number of the stored documents is big. So to avoid this issues we will retrive only the most three siomilar documents from the vector Data Base.

## Vectore data base

The vectore data base that we are using is pinecone, which is a user freindly tool, in which all we have to do is creating an account, then we can use the default project already created for us in order to create an index that we will use to store the representations of the documents.
	![ alt text for screen readers](/images/d8e002f5074a908faee547fc24a48e77dec727c4.png)

## Project workflow

in order to build this project, we started by bulding a user interface by streamlit, a python library that makes it very simple to build web applications for similar objectives to ours.

the first step that users need to do is iploading a pdf file, that's why we used pypdf2 which is is a very poerful tool to extract text from PDF files.

After that we will use a t5 architecture model to generate questions  based on the extracted text, and we will use haystack in order to firstlly retrieve the most three similar documents on our vector Data Base. After retriving the documents we use a bart architecture in order to generate the question.
	![ alt text for screen readers](/images/workflow.png)

## Files descreption

1. Tools and functionalities:
			In this file you will find a group of tools and functionaliteis that I developed and will be developing since I am still working on this prject.

2. pipelines.py
			In this file you will find the tools used for the end to end question generation.

3. alpha.py
		This file represnts our user intrface, you will find in it the functionalities and also the streamlit code.

4. requirments
		This file containes the libraries that you should install in order to run this application.

## How to excute the app

In order to excute the app all you have to do is

- Create an account in pincone in order to get you api_key and index.
- install the needed libraries (Already mentioned in requirments).
- download this repository, make sure your prompt is inside it and excute this command

## command

		streamlit run alpha.py

## pics from the app

![ alt text for screen readers](/images/1.png)
![ alt text for screen readers](/images/2.png)
![ alt text for screen readers](/images/3.png)
![ alt text for screen readers](/images/4.png)


