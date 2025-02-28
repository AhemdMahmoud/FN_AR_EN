# IMDb and Arabic Sentiment Analysis with DistilBERT and M-BERT

## Overview
This project demonstrates domain adaptation by fine-tuning DistilBERT on the Large Movie Review Dataset (IMDb) and Modern BERT on an Arabic dataset from FineWeb 2. The goal is to adapt the models' vocabulary from factual Wikipedia data (used in pretraining) to subjective movie and text reviews. The fine-tuned models can then be used for sentiment analysis.

## Dataset
### IMDb Dataset
We use the **Large Movie Review Dataset (IMDb)**, a well-known corpus for benchmarking sentiment analysis models. The dataset is available on the Hugging Face Hub and can be accessed using the `load_dataset()` function from the 🤗 Datasets library.

### Arabic Dataset (FineWeb 2)
For Arabic sentiment analysis, we fine-tune **Modern BERT (M-BERT)** on an Arabic dataset sourced from **FineWeb 2**, a high-quality web-scraped Arabic corpus. This dataset enables the model to adapt to various Arabic dialects and formal text.

## Model
The project leverages:
- **DistilBERT**: A smaller, faster, and cheaper variant of BERT, designed to maintain high performance while being computationally efficient.
- **Modern BERT (M-BERT)**: A transformer model specifically trained for Arabic text, enabling better sentiment analysis on Arabic datasets.

The model checkpoints used:

```python
from transformers import AutoModelForMaskedLM

# English model (IMDb)
model_checkpoint_en = "distilbert-base-uncased"
model_en = AutoModelForMaskedLM.from_pretrained(model_checkpoint_en)

# Arabic model (FineWeb 2)
model_checkpoint_ar = "aubmindlab/bert-base-arabertv02"
model_ar = AutoModelForMaskedLM.from_pretrained(model_checkpoint_ar)
```

We can check the number of parameters in each model using:

```python
print("DistilBERT Parameters:", model_en.num_parameters())
print("Modern BERT Parameters:", model_ar.num_parameters())
```

## Training Procedure
The fine-tuning process includes training the models on respective datasets and evaluating performance using perplexity as a metric.

### Training Setup
The training setup includes:
- Using the `Trainer` API from Hugging Face's `transformers` library
- Downsampling the dataset for faster training
- Utilizing a `data_collator` for batch processing

```python
from transformers import Trainer

trainer_en = Trainer(
    model=model_en,
    args=training_args,
    train_dataset=downsampled_dataset_en["train"],
    eval_dataset=downsampled_dataset_en["test"],
    data_collator=data_collator,
    tokenizer=tokenizer_en,
)

trainer_ar = Trainer(
    model=model_ar,
    args=training_args,
    train_dataset=downsampled_dataset_ar["train"],
    eval_dataset=downsampled_dataset_ar["test"],
    data_collator=data_collator,
    tokenizer=tokenizer_ar,
)
```

### Training Loop
The training loop involves multiple epochs, with gradient accumulation and learning rate scheduling. We use `tqdm` for a progress bar and `torch` for loss calculations.

```python
from tqdm.auto import tqdm
import torch
import math

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    for model, train_dataloader, eval_dataloader, tokenizer, dataset in [
        (model_en, train_dataloader_en, eval_dataloader_en, tokenizer_en, eval_dataset_en),
        (model_ar, train_dataloader_ar, eval_dataloader_ar, tokenizer_ar, eval_dataset_ar)
    ]:
        # Training
        model.train()
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluation
        model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(batch_size)))

        losses = torch.cat(losses)
        losses = losses[: len(dataset)]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

        print(f">>> Epoch {epoch} ({'English' if model == model_en else 'Arabic'}): Perplexity: {perplexity}")
```

### Model Saving and Uploading
After each epoch, we save and upload the trained models to the Hugging Face Hub.

```python
# Save and upload
for model, tokenizer, output_dir in [
    (model_en, tokenizer_en, output_dir_en),
    (model_ar, tokenizer_ar, output_dir_ar)
]:
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(
            commit_message=f"Training in progress epoch {epoch}", blocking=False
        )
```

## Requirements
To run this project, install the necessary dependencies:

```bash
pip install transformers datasets accelerate torch tqdm
```

## Usage
1. Load the IMDb and FineWeb 2 datasets using `datasets.load_dataset()`.
2. Initialize the models and tokenizers.
3. Fine-tune the models using the `Trainer` API or custom training loop.
4. Evaluate the model performance using perplexity.
5. Save and upload the fine-tuned models.

## Conclusion
This project demonstrates the domain adaptation of DistilBERT for sentiment analysis on IMDb reviews and Modern BERT for Arabic sentiment analysis using FineWeb 2. The fine-tuned models adapt to subjective text and are valuable tools for analyzing sentiment in English and Arabic.

