# Week 12 Overview
This week, we will continue both our information retrieval and large language model (LLM) coverage with a deep dive into LLM embedding, fine-tuning, and general post-training. We will start with a look at the training process of large models and how the common pre- and post-training phases drive critical capabilities and enable customization. We will continue with applications using the internal embeddings of these models as convenient document vectors for classification, search, and more reliable generation. The first topic takes existing language models and makes them better behaved. The second topic takes existing models and adapts them to other applications. This week's project will revisit the previous email spam classification problem using language models.

## Learning Objectives 
At the end of this week, you will be able to: 
- Explain the relationship between pre-training and post-training. 
- Explain the benefits of post-training for language models. 
- Explain the benefits of low-rank adaptations for model fine-tuning. 
- Apply latent codes and embeddings from pre-trained models to classification tasks. 
- Apply latent codes and embeddings to document search tasks. 
- Apply language model embeddings for retrieval-augmented generation.

## Topic Overview: Post-training Large Models
Training large models in the recent big data era is frequently a tradeoff between quantity and quality. Large language or image models are typically trained on as much freely available material on the internet as can be acquired. This gives a lot of data to work with and helps train large neural networks that generalize. 

However, the typical weaknesses are twofold. First, the quality of the input data is hard to maintain while collecting as much existing content as available. Second, this input data is rarely labeled, so these large training models are limited to unsupervised learning. 

Together, these weaknesses limit our largest training processes to reproducing the distributions of data as it already existed when we gathered it. However, our goals for using these models tend to be loftier. We do not want to reproduce the existing mediocrity but exceed it. To do so, we follow up the large training process, often called "pre-training," with one or more "post-training" phases that improve the model with smaller datasets that are higher quality or labeled.

## 1.1 Lesson: Training Large Language Models in Stages
Last week, we discussed the core functionality of large language models, next token prediction, but we said little about how such a model is trained. We will now look into that process more, as variations in that training process substantially change our evaluations and applications of those models.

### Stages of Training a Large Language Model
Best practices in training large language models continue to evolve, but the typical consists of pre-training on as large a clean dataset as possible and post-training to shape its output. Intuitively, the pre-training acquires broad knowledge from the large dataset while the post-training guides how that knowledge is used and expressed.

#### Pre-training
*Pre-training is an initial model training process, typically unsupervised learning, that is expected to facilitate further training with new datasets or objectives.*

Most large language models are pre-trained on the next token prediction task. Many image classification models were pre-trained on large image datasets such as ImageNet. 

#### Post-training
*Post-training is a later model training process that further trains model weights after pre-training.*

This term has been used much more following recent large language model successes.

There are many variations of post-training, and some are used in combination with each other. Here are some of the most common examples.

#### Supervised Fine-tuning
*Supervised fine-tuning trains a pre-trained model with a supervised learning task.*

For language models, supervised fine-tuning typically consists of 10,000 to 100,000 examples of prompts and desired responses. The same next token prediction task is used, but the model is only optimized to produce the desired response. It does not need to learn to produce the prompts. 


#### Instruction Fine-tuning
*Instruction fine-tuning is a supervised fine-tuning process aimed at making a model follow instructions well.*

Instruction fine-tuning does not necessarily use supervised learning. There were a few attempts at making language models follow instructions reliably. The first with broad success and deployment was InstructGPT (OpenAI, 2022). 

#### Reinforcement Learning from Human Feedback (RLHF)
*Reinforcement learning from human feedback trains a reward model to learn which language responses are favored by humans and then uses that reward model to fine-tune the language model.*

InstructGPT used RLHF to improve instruction following in the OpenAI API and was an immediate predecessor to the first ChatGPT model. A challenge with RLHF is making responses favored by humans without straying too far from the knowledge originally encoded during pre-training. That is, how can understanding of human preferences be integrated without overwriting the knowledge learned during pre-training? For example, a preference for polite responses should not interfere with understanding impolite questions or rough language.

### Post-training Mechanics
Most post-training steps will train the language model in the same way, with more specialized data and even the same loss function. This is generally the case for post-training steps described as fine-tuning. Post-training steps, such as reinforcement learning from human feedback with more radically different targets, will use hybrid objective functions to adjust the language model without changing its output too much.

## 1.2 Lesson: Low-rank Adaptations of Pre-trained Models
Training a large language model tends to be an expensive task. Nowadays, 100s or 1,000s of commercial-grade graphics processing units (GPUs) are required to train a state-of-the-art language model. Few parties have the resources to train a bespoke language model, so many companies and researchers have considered fine-tuning language models created by other parties. This process is still expensive, though; a full fine-tuning of a model creates a new copy of the large language model of the same size. To reduce these costs, low-rank adaptations of pre-trained models were proposed as a cheaper alternative to full model fine-tuning.

### Further Motivation for Fine Tuning
There are many reasons to fine-tune a model. One reason is post-training. Another is to provide better handling for cases not well-covered by the training data. Another might be to integrate proprietary data not available to the original training process. Or to personalize responses to an individual user's preferences.

### Motivation for Cheaper Fine-tuning
Most of the fine-tuning use cases above are desirable to companies and people without the resources to train a model from scratch. Even fine-tuning all the parameters of a state-of-the-art language model may be impractical given the memory and storage requirements. GPT-3.5 had approximately 175 billion parameters, and OpenAI offered model fine-tuning as a service to API customers, but even Microsoft had trouble implementing that model fine-tuning process (Hu et al., 2021). Those 175 billion parameters required more than 1TB of disk space in their original uncompressed form. Just the GPUs required to make predictions with such a model are out of the budget of small companies and individuals.

### Low-rank adaptations of Text
The basic idea of low-rank adaptations is simple (Hu et al., 2021).

Instead of allowing arbitrary updates to large-parameter $n \times n$ matrices $M$ in a model, only allow low-rank updates. 

That is, if $M^{\prime}$ is the updated version of $M$, restrict $M^{\prime}$ fo have the form $M^{\prime} \; = \; M + AB$, where:
- $A$ is an $n \times r$ matrix, 
- $B$ is an $r \times n$ matrix, 
- and $r$ is much lower than $n$

The additional cost to store $A$ and $B$ is only $2nr$ weights, which will be less than storing a complete copy $M^{\prime}$ with its $n^2$ parameters as long as $r < n/2$. During the fine-tuning process, the gradients only need to be computed for $A$ and $B$ instead of $M^{\prime}$, substantially decreasing the computation and memory needed for backpropagation.

![alt text](<images/Week 12 Low Rank Adaptations V2.png>)

### Low-rank Adaptations of Images
The idea of low-rank adaptations (LoRA) is more general than just language models. The original paper (Hu et al., 2021) did not mention images at all, but the budding generative image communities at the time quickly made the connection and started making LoRA adaptations to favor specific art styles and characters. Custom LoRA generations are a popular offering on freelancer and artificial intelligence (AI) art sites to this day.

# Topic Overview: Adapting Models to New Applications
Language models fall under the umbrella label "generative AI;" language models are designed and trained for the specific task of generating content. Similarly, there are image generation models designed and trained for creating new images, and these models and their use are also labeled as generative AI. Both language models and image models have existed for decades but have dramatically improved in recent years. With these improvements, these models have developed more and more useful computations supporting realistic text or image generation. And it turns out these computations are useful for many more tasks besides content generation. In this topic, we will investigate applying existing models to new tasks and applications.

Here’s a situation in which this expanded application may be beneficial.

## 2.1 Lesson: Building Classifiers from Generative Models
The idea of using the representations from generative models for other tasks is not new. Turing Award and Nobel Prize recipient Geoffrey Hinton has said for many years that computer vision should be considered akin to inverse computer graphics; being able to draw a scene suggests understanding what was portrayed in the scene. Hinton's research (previously covered in Module 1) used small latent codes from auto-encoder models to classify images (Hinton et al., 2006). We will now review that work and more recent results.

### Building Classifiers from Auto-encoder Latents
The idea of using small latent codes as simpler model inputs dates back to at least 1957, when principal components analysis was recommended as a source of better model inputs by both Hostelling (1957) and Kendall (1957). Here, the analysis extracted “latent codes,” small vectors that could be used to approximately reconstruct the input. The original latent codes were coordinates from principal components analysis.

Hinton's work extended this approach to latent codes computed with neural networks as a form of "non-linear principal components analysis" where an auto-encoder network was trained to reconstruct images from small latent codes between the encoder and decoder components of the auto-encoder (Hinton et al., 2006). Further tests with those codes found that simple linear classifiers were effective for labeling the modeled images.

An important point to note is that the latent codes can be computed or trained without access to image labels. The latent codes are a reflection of the distribution of images. This allows the latent codes to be trained from a large unlabeled dataset, and then the classifier can be trained from a much smaller labeled dataset. Linear classifiers in particular do not tend to need many more samples than the number of dimensions, and latent codes will have dimensions much smaller than the size of the dataset used to train them. Since labeled datasets tend to require human labeling and be more expensive, splitting the process into using a large unlabeled dataset to train an auto-encoder for latent codes and using a small labeled dataset to train the classifier can be a large cost savings.

### Building Text Classifiers from Language Model Embeddings
A popular pre-trained language model for classification tasks is **BERT (bidirectional encoder representations from transformers)** (Devlin et al., 2019).

**BERT** is a language model based on attention mechanisms and the transformer architecture like most recent language models (covered in Module 7), but it was trained on a masked token prediction task instead of next token prediction. 

That is, BERT is trained to fill in the blanks if words are masked out from a piece of text. BERT uses a special [CLS] token to mark the beginning of text, and the final attention outputs for this [CLS] token are used as a "text embedding" vector. 

This embedding vector is not designed to recover the original text like a latent code. But since it is part of the state used to predict masked tokens, it encodes much of the surrounding context, which would be used for those masked tokens. By design, this makes the embedding vector very suitable as input for classification problems.

The BERT paper tested two variants of their model, dubbed $\mathbf{BERT_{\text{BASE}}}$ and $\mathbf{BERT_{\text{LARGE}}}$ with hidden vectors of length 768 and 1,024, respectively. 

After pre-training the BERT model on the masked token prediction task, they added a linear layer mapping from the final hidden vector for 
the [CLS] token to outputs for each class. 

For a classification problem with $k$ classes, this linear layer would be implemented by a matrix of size $768 \times k$ or $1024 \times k$ depending on the model used. They tested two ways of training the resulting classification model:

- The first approach, which they called “full fine-tuning," trained all the parameters of the original BERT model plus the new linear layer. 
- The second approach was described as "feature-based" and only trained the new linear layer. This latter feature-based approach treated the final hidden vector as a fixed set of features that could be shared across multiple classification problems. 

The full fine-tuning generally had higher classification performance, but the feature-based approach was generally close behind and often beat pre-BERT approaches. The feature-based approach additionally had the advantage of requiring a lot less incremental storage for each new classification problem, much faster training, and generally lower requirements for labeled data (a few thousand samples is often enough in practice).

Language models have improved substantially in the time since BERT was developed, but BERT is still popular for classification since its small size allows it to be run more easily with minimal or no GPU resources. ($\mathbf{BERT_{\text{LARGE}}}$ has a mere 340 million parameters, while recent large models are approximately 1,000 times larger.) Additionally, BERT embeddings are often sufficient for accurate classifiers. That said, most current language models also support extracting embeddings for feature-based classification, and they may perform better for more subtle classification tasks.

### Building Image Classifiers from Generative Image Models
As previously mentioned, some of the earliest uses of latent codes for classification were for images. Recent work has been very focused on generation capabilities, but occasional checks still find the latent codes to be useful in analyzing the images. A recent example by Skorokhodov et al. (2021) building a model INR-GAN to output continuous images (rather than a grid of pixels) found that their latent codes from a model trained on pictures of faces were able to accurately estimate locations of facial keypoints using just a linear model. They speculated that their INR-GAN's focus on predicting color based on location improved its spatial awareness and ability to predict locations compared to previous work such as StyleGAN, which did not explicitly consider coordinates.

![alt text](<images/Week 12 geometric-prior.jpg>)

### Building Classifiers from Generative Models

## 2.2 Lesson: Indexing Documents with Language Model Embeddings
Indexing documents with document vectors was previously discussed in Week 10. Lesson 2.1 just discussed using language model embeddings as feature vectors for feature-based classification models. Can these language model embeddings also be used as document vectors for search?

### Describing Documents with Language Model Embeddings
Previously, Peter observed that identifying relevant documents is as difficult as answering questions based on internal company documents. Can a large language model pre-read the internal documents and make searching easier? 

It turns out that language model embeddings are great document vectors. Most documents will fit within the token limits of recent language models, so it is possible to compute and store document vectors for every document. A first pass at indexing documents likely computes the language model embedding for every document and stores them in a vector database. This is often a very good pass, but there may still be easy improvements.

The structures of the documents may suggest high-value excerpts or individual paragraphs that it might be valuable to refer to directly. For example, a document abstract might be worth indexing individually. A long document might be better indexed with a collection of paragraph vectors, each of which is the language model's embedding of the paragraph. Such embeddings might allow more specific search results referring to exactly the paragraph where a question is answered. However, embedding paragraph vectors will increase the size of the vector database specifically, and only embedding one paragraph at a time may reduce the relevant context included in the embedding. Whether this context is worthwhile will depend on the specific application and documents.

### Implementing Document Search with Language Model Embeddings
(03:20)

Language model embeddings are very useful keys for document search and generally provide better search than the document vectors previously used in Week 10. In this video, we use language model embeddings to find similar recipes and to search for recipes given a description. 

## 2.3 Lesson: Retrieval-augmented Generation
A persistent problem with language model output is "hallucinations," outputs where it appears the model just made up an incorrect answer. A secondary problem is that the information of the language model is out of date; the information requested has changed since the training data was collected. Both of these problems can be reduced by providing context containing the right answer, but isn't providing the answer in the prompt missing the point of having the model answer the question? That is the case if the content providing the answer was chosen manually. Using retrieval-augmented generation (RAG), documents providing relevant context are automatically identified and retrieved using language model embeddings as described in Lesson 2.2, and that context is automatically added to the prompt.

### Referencing Documents with Language Model Embeddings
A prerequisite for searching for relevant documents with a vector database is being able to construct a query vector that is similar to the document vectors that we wish to find. Fortunately, language model embeddings tend to have similar embeddings for questions and answers. So, given a prompt that needs supporting documents for context, we can use the language model embedding of the prompt to search for related documents.

### Retrieval-augmented Generation
The idea of RAG is that a prompt can be processed to produce a query embedding, that query embedding can be used to look up related documents, and those documents can be used to generate a response (Lewis et al., 2020). The details have changed significantly since the original RAG proposal. Previously, there were separate encoders for questions and documents. Nowadays, the same language models are typically used for both embeddings, and matching documents are copied wholesale into the language model prompts. 

