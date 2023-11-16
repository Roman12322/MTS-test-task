# MTS-test-task

## Task description

Goal: fine-tune model with less than 7b parameters for buying air tickets on a custom/found dataset.  

## Dataset overview:

As for dataset i chose [bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/blob/main/README.md), because
this dataset can be used to train Large Language Models such as GPT, Llama2 and Falcon, both for Fine Tuning and Domain Adaptation.

The dataset has the following specs:

- Use Case: Intent Detection
- Vertical: Customer Service
- 27 intents assigned to 10 categories
- 26872 question/answer pairs, around 1000 per intent
- 30 entity/slot types
- 12 different types of language generation tags

The categories and intents have been selected from Bitext's collection of 20 vertical-specific datasets, covering the intents that are common across all 20 verticals. The verticals are:

- Automotive, Retail Banking, Education, Events & Ticketing, Field Services, Healthcare, Hospitality, Insurance, Legal Services, Manufacturing, Media Streaming, Mortgages & Loans, Moving & Storage, Real Estate/Construction, Restaurant & Bar Chains, Retail/E-commerce, Telecommunications, Travel, Utilities, Wealth Management

For a full list of verticals and its intents see [https://www.bitext.com/chatbot-verticals/](https://www.bitext.com/chatbot-verticals/).

The question/answer pairs have been generated using a hybrid methodology that uses natural texts as source text, NLP technology to extract seeds from these texts, and NLG technology to expand the seed texts. All steps in the process are curated by computational linguists.

### Dataset Token Count

The dataset contains an extensive amount of text data across its 'instruction' and 'response' columns. After processing and tokenizing the dataset, we've identified a total of 3.57 million tokens. This rich set of tokens is essential for training advanced LLMs for AI Conversational, AI Generative, and Question and Answering (Q&A) models.

### Fields of the Dataset

Each entry in the dataset contains the following fields:

- flags: tags (explained below in the Language Generation Tags section)
- instruction: a user request from the Customer Service domain
- category: the high-level semantic category for the intent
- intent: the intent corresponding to the user instruction
- response: an example expected response from the virtual assistant

### Categories and Intents

The categories and intents covered by the dataset are:

- ACCOUNT: create_account, delete_account, edit_account, switch_account
- CANCELLATION_FEE: check_cancellation_fee
- DELIVERY: delivery_options
- FEEDBACK: complaint, review
- INVOICE: check_invoice, get_invoice
- NEWSLETTER: newsletter_subscription
- ORDER: cancel_order, change_order, place_order
- PAYMENT: check_payment_methods, payment_issue
- REFUND: check_refund_policy, track_refund
- SHIPPING_ADDRESS: change_shipping_address, set_up_shipping_address

## Model
The model ybelkada/falcon-7b-sharded-bf16 was chosen for training. The following arguments were put forward to justify the choice: 
- Predisposition: the developers of the model claim that it is more prone to fine-tuning on a dataset that implies excellent interaction in the User-Assistant system.
- Why not Llama-7b? Llama-7b has proven its accuracy, however, we wanted to experiment and find an analogue that, all other things being equal, is also able to accurately perform the tasks set in the examples of User-Assistant systems.

## Fine-tuning
For fine-tuning model was used specific LoraConfig. You may wanna play with lora_dropout, lora_r, lora_alpha as well and adjust it more precisely.
```
lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]
)
```
## Metrics
To track your scores, you can use BLEU-score, it really helps to track the question answering pattern and provide more relevant answers. 
### How does that work:
Imagine that your question like: "how can i get tickets from Moscow to Tokyo?". Given that question your expected answer would be like: "You need to look up some tickets on [website of aggregator or flying company] and buy some", but there're nuances that you have to be aware of:
- Fine-tuned model may be a little bit off the right topic (talking about trains, not airplanes)
- Your question may be not quite well organized
Assuming all these nuances we can calculate how our answer was relevant/not relevant.
[Link to BLEU](https://www.digitalocean.com/community/tutorials/bleu-score-in-python)
