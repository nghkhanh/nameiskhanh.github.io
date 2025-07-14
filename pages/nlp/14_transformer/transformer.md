---
layout: default
title: 14. Transformer
nav_order: 14
parent: NLP
permalink: /nlp/14_transformer
---

# Transformer

## Introduction

Transformers are a groundbreaking neural network architecture that revolutionized natural language processing. Unlike traditional models, they rely entirely on self-attention mechanisms, allowing them to handle long-range dependencies more efficiently. 

In this lesson, we'll explore how Transformers work, their key components, and implematation of Transformer architecture in summarization task.

## Transformer architecture
The Tranformer architecture consists of an encoder and a decoder, both composed of multiple layers of self-attention mechanisms and feedforward neural networks. Self-attention allows the model to weigh the importance of different words in a sequence, capturing contextual information effectively. This architecture is designed for sequential data processing, such as language translation and text generation.

![](draw/TransformerArchitecture.png)

We will learn Transformer Encoder and Transformer Decoder in turn.

## Transformer Encoder
### Encoder components
![](draw/EncoderComponents.png)

### Encoder workflow
#### Word Embeddings
Similar to RNN, Transformer also uses Embedding to represent words by vectors.
![](draw/Step1.png)

#### Positional Encodings
Because Transformer architecture does not have recurrence or convolution, to ensure the order of the sequence, we have to add information about the absolute and relative position of tokens in the sequence. 
![](draw/Step2.png)

We have two methods to build **Positional Encoding**, there are learnable positional embedding and pre-defined function. We will only mention to learnable positional embedding method because it is more popular than pre-defined function method. 

![](draw/LeanableEmbedding.png)

#### Multi-Headed Attention
Implement Multi-Headed Attention to enhance representation of embeddings.

**Note**: To learn more about Multi-Headed Attention, let's take a look at [this link](https://github.com/robusto-ai/elearning-learning-materials/blob/dunghc/transformer/Nlp/Attention/attention.md)

![](draw/Multiheaded.png)

#### Add & Norm

![](draw/AddNorm.png)

Besides using Layer Norm, we also apply a residuals technique. This not only helps **Gradient Flow** be better, preventing vanishing gradient when training deep neuron networks, but also brings positional information from lower layer to higher layer.

#### Position-wise Feed-Forward Network

![](draw/Pointwise.png)

#### Next step
The results from **Add & Norm** will be fed into another encoder layer (which comprises of Multi-Head Attention and Position-wise Fead-Forward Network) until reach to the number of defined encoder layers.

## Transformer Decoder

![](draw/Decoder.png)

### Decoder components

![](draw/DecoderComponent.png)

#### Embeddings
Just like Encoder, we also have to combine output Word Embeddings and Positional Embeddings before feeding to Multi-Headed Attention.

![](draw/DecoderEmbedding.png)

#### Decoder Training and Inference
Depend on training or inference, Decoder will behave differently. 

#### Decoder Traning
![](draw/DecoderTraining.png)

#### Decoder Inference
![](draw/DecoderInference1.png)

![](draw/DecoderInference2.png)

![](draw/DecoderInference3.png)

![](draw/DecoderInference4.png)

![](draw/DecoderInference5.png)

#### Decoder Problem with Self Attention
![](draw/AttentionProblem.png)

#### Masked Multi-Headed Attention
Generally, it is quite similar to Multi-Headed Attention but it has additional component called **Mask**.

![](draw/Mask1.png)

![](draw/Mask2.png)

#### Decoder Multi-Headed Attention

![](draw/DecoderMultiheadedAttention.png)

#### Decoder Multi-Headed Attention With Encoder

![](draw/DecoderMultiheadedAttentionEncoder.png)

#### Linear Classifier

![](draw/DecoderLinear.png)

## Implementing Transformer model
We have talk a lot, it's time to write some scripts. In this session, we will use pretrained T5, which is an variant of Transformer, to solve translation task. We will use **HuggingFace** library for easy implementation.

### Install HuggingFace libraries's ecosystem
```python
pip install transformers[torch] datasets evaluate rouge_score
```

### Import necessary libraries
```python
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import create_optimizer, AdamWeightDecay
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
```

### Load BillSum dataset
```python
billsum = load_dataset("billsum", split="ca_test")
```

### Split Train, Test
```python
billsum = billsum.train_test_split(test_size=0.2)
billsum["train"][0]

{'summary': 'Existing law authorizes state agencies to enter into contracts for the acquisition of goods or services upon approval by the Department of General Services. Existing law sets forth various requirements and prohibitions for those contracts, including, but not limited to, a prohibition on entering into contracts for the acquisition of goods or services of $100,000 or more with a contractor that discriminates between spouses and domestic partners or same-sex and different-sex couples in the provision of benefits. Existing law provides that a contract entered into in violation of those requirements and prohibitions is void and authorizes the state or any person acting on behalf of the state to bring a civil action seeking a determination that a contract is in violation and therefore void. Under existing law, a willful violation of those requirements and prohibitions is a misdemeanor.\nThis bill would also prohibit a state agency from entering into contracts for the acquisition of goods or services of $100,000 or more with a contractor that discriminates between employees on the basis of gender identity in the provision of benefits, as specified. By expanding the scope of a crime, this bill would impose a state-mandated local program.\nThe California Constitution requires the state to reimburse local agencies and school districts for certain costs mandated by the state. Statutory provisions establish procedures for making that reimbursement.\nThis bill would provide that no reimbursement is required by this act for a specified reason.',
 'text': 'The people of the State of California do enact as follows:\n\n\nSECTION 1.\nSection 10295.35 is added to the Public Contract Code, to read:\n10295.35.\n(a) (1) Notwithstanding any other law, a state agency shall not enter into any contract for the acquisition of goods or services in the amount of one hundred thousand dollars ($100,000) or more with a contractor that, in the provision of benefits, discriminates between employees on the basis of an employee’s or dependent’s actual or perceived gender identity, including, but not limited to, the employee’s or dependent’s identification as transgender.\n(2) For purposes of this section, “contract” includes contracts with a cumulative amount of one hundred thousand dollars ($100,000) or more per contractor in each fiscal year.\n(3) For purposes of this section, an employee health plan is discriminatory if the plan is not consistent with Section 1365.5 of the Health and Safety Code and Section 10140 of the Insurance Code.\n(4) The requirements of this section shall apply only to those portions of a contractor’s operations that occur under any of the following conditions:\n(A) Within the state.\n(B) On real property outside the state if the property is owned by the state or if the state has a right to occupy the property, and if the contractor’s presence at that location is connected to a contract with the state.\n(C) Elsewhere in the United States where work related to a state contract is being performed.\n(b) Contractors shall treat as confidential, to the maximum extent allowed by law or by the requirement of the contractor’s insurance provider, any request by an employee or applicant for employment benefits or any documentation of eligibility for benefits submitted by an employee or applicant for employment.\n(c) After taking all reasonable measures to find a contractor that complies with this section, as determined by the state agency, the requirements of this section may be waived under any of the following circumstances:\n(1) There is only one prospective contractor willing to enter into a specific contract with the state agency.\n(2) The contract is necessary to respond to an emergency, as determined by the state agency, that endangers the public health, welfare, or safety, or the contract is necessary for the provision of essential services, and no entity that complies with the requirements of this section capable of responding to the emergency is immediately available.\n(3) The requirements of this section violate, or are inconsistent with, the terms or conditions of a grant, subvention, or agreement, if the agency has made a good faith attempt to change the terms or conditions of any grant, subvention, or agreement to authorize application of this section.\n(4) The contractor is providing wholesale or bulk water, power, or natural gas, the conveyance or transmission of the same, or ancillary services, as required for ensuring reliable services in accordance with good utility practice, if the purchase of the same cannot practically be accomplished through the standard competitive bidding procedures and the contractor is not providing direct retail services to end users.\n(d) (1) A contractor shall not be deemed to discriminate in the provision of benefits if the contractor, in providing the benefits, pays the actual costs incurred in obtaining the benefit.\n(2) If a contractor is unable to provide a certain benefit, despite taking reasonable measures to do so, the contractor shall not be deemed to discriminate in the provision of benefits.\n(e) (1) Every contract subject to this chapter shall contain a statement by which the contractor certifies that the contractor is in compliance with this section.\n(2) The department or other contracting agency shall enforce this section pursuant to its existing enforcement powers.\n(3) (A) If a contractor falsely certifies that it is in compliance with this section, the contract with that contractor shall be subject to Article 9 (commencing with Section 10420), unless, within a time period specified by the department or other contracting agency, the contractor provides to the department or agency proof that it has complied, or is in the process of complying, with this section.\n(B) The application of the remedies or penalties contained in Article 9 (commencing with Section 10420) to a contract subject to this chapter shall not preclude the application of any existing remedies otherwise available to the department or other contracting agency under its existing enforcement powers.\n(f) Nothing in this section is intended to regulate the contracting practices of any local jurisdiction.\n(g) This section shall be construed so as not to conflict with applicable federal laws, rules, or regulations. In the event that a court or agency of competent jurisdiction holds that federal law, rule, or regulation invalidates any clause, sentence, paragraph, or section of this code or the application thereof to any person or circumstances, it is the intent of the state that the court or agency sever that clause, sentence, paragraph, or section so that the remainder of this section shall remain in effect.\nSEC. 2.\nSection 10295.35 of the Public Contract Code shall not be construed to create any new enforcement authority or responsibility in the Department of General Services or any other contracting agency.\nSEC. 3.\nNo reimbursement is required by this act pursuant to Section 6 of Article XIII\u2009B of the California Constitution because the only costs that may be incurred by a local agency or school district will be incurred because this act creates a new crime or infraction, eliminates a crime or infraction, or changes the penalty for a crime or infraction, within the meaning of Section 17556 of the Government Code, or changes the definition of a crime within the meaning of Section 6 of Article XIII\u2009B of the California Constitution.',
 'title': 'An act to add Section 10295.35 to the Public Contract Code, relating to public contracts.'}
```

### Preprocess
```python
checkpoint = "google-t5/t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

prefix = "summarize: "
def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_billsum = billsum.map(preprocess_function, batched=True)
```

### Initialize data collator
This component will gather many samples into one batch.
```python
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
```

### Define evaluate metrics
```python
rouge = evaluate.load("rouge")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}
```
### Train model using Trainer class of HuggingFace library
We need GPU to train this kind of model faster.
#### Define training arguments
```python
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

training_args = Seq2SeqTrainingArguments(
    output_dir="my_awesome_billsum_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=4,
    predict_with_generate=True,
    fp16=True,
)
```

#### Train model
```python
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_billsum["train"],
    eval_dataset=tokenized_billsum["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```

### Finetune model with more customization (optional)
#### Define optimizer
```python
optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
```

#### Load train, test data
```python
tf_train_set = model.prepare_tf_dataset(
    tokenized_billsum["train"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

tf_test_set = model.prepare_tf_dataset(
    tokenized_billsum["test"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)
```

#### Finetune model 
```python
model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)
model.compile(optimizer=optimizer)

# Define callback
callbacks = [KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_test_set)]

# Train model
model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3, callbacks=callbacks)
```

#### Save model 
```python
trainer.save_model("my_summarization_model")
```

### Inference
```python
text = "summarize: The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."
summarizer = pipeline("summarization", model="my_summarization_model")
summarizer(text)

[{"summary_text": "The Inflation Reduction Act lowers prescription drug costs, health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, which will lift up American workers and create good-paying, union jobs across the country."}]
```

## Conclusion

In conclusion, the lesson on Transformers highlighted how this innovative architecture leverages self-attention mechanisms to efficiently handle long-range dependencies in data. We've seen how Transformers form the backbone of advanced models like BERT and GPT, driving breakthroughs in natural language processing.

## References

+ J. Alammar, “The Illustrated Transformer,” jalammar.github.io, Jun. 27, 2018. https://jalammar.github.io/illustrated-transformer/
+ “Summarization,” huggingface.co. https://huggingface.co/docs/transformers/tasks/summarization



