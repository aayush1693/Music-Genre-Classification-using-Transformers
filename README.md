# Music Genre Classification using Transformers

This project demonstrates the use of transformers for classifying music genres. It leverages the DistilHuBERT model and the GTZAN dataset to achieve high accuracy in genre classification.

## Installation

To install the required modules, run the following commands:

```sh
pip install transformers
pip install accelerate
pip install datasets
pip install evaluate
```

## Usage

### Importing Required Libraries

```python
from datasets import load_dataset, Audio
import numpy as np
from transformers import pipeline, AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer
import evaluate
```

### Loading and Splitting the Dataset

```python
gtzan = load_dataset("marsyas/gtzan", "all")
gtzan = gtzan["train"].train_test_split(seed=42, shuffle=True, test_size=0.1)
```

### Data Pre-processing

```python
model_id = "ntu-spml/distilhubert"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True, return_attention_mask=True)
sampling_rate = feature_extractor.sampling_rate
gtzan = gtzan.cast_column("audio", Audio(sampling_rate=sampling_rate))

def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * 20.0),
        truncation=True,
        return_attention_mask=True,
    )
    return inputs

gtzan_encoded = gtzan.map(preprocess_function, remove_columns=["audio", "file"], batched=True, batch_size=25, num_proc=1)
gtzan_encoded = gtzan_encoded.rename_column("genre", "label")
```

### Classification Model

```python
model = AutoModelForAudioClassification.from_pretrained(model_id, num_labels=num_labels, label2id=label2id, id2label=id2label)

training_args = TrainingArguments(
    f"{model_name}-Music classification Finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    warmup_ratio=0.1,
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
)
```

### Model Evaluation

```python
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

trainer = Trainer(
    model,
    training_args,
    train_dataset=gtzan_encoded["train"],
    eval_dataset=gtzan_encoded["test"],
    tokenizer=feature_extractor,
    compute_metrics=compute_metrics,
)
trainer.train()
```

### Loading and Saving the Model

```python
model.save_pretrained("/content/Saved Model")
feature_extractor.save_pretrained("/content/Saved Model")

loaded_model = AutoModelForAudioClassification.from_pretrained("/content/Saved Model")
loaded_feature_extractor = AutoFeatureExtractor.from_pretrained("/content/Saved Model")
```

### Pipeline for Classification

```python
pipe = pipeline("audio-classification", model=loaded_model, feature_extractor=loaded_feature_extractor)

def classify_audio(filepath):
    preds = pipe(filepath)
    outputs = {}
    for p in preds:
        outputs[p["label"]] = p["score"]
    return outputs

input_file_path = input('Input:')
output = classify_audio(input_file_path)

print("Predicted Genre:")
max_key = max(output, key=output.get)
print("The predicted genre is:", max_key)
print("The prediction score is:", output[max_key])
```

## Conclusion

Music genre classification presents a complex and computationally intensive challenge with broad applications across various industries. The implemented model, leveraging a DistilHuBERT-based architecture and fine-tuned on the GTZAN dataset, achieved a respectable accuracy. Further performance enhancements could be explored by utilizing a larger and more diverse dataset to improve the model's generalization capabilities and potentially achieve even higher accuracy.

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/aayush1693/Music-Genre-Classification-using-Transformers/blob/main/LICENSE) file for details.
