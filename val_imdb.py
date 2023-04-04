import torch
from torch import nn
import numpy as np
import pandas as pd 

# BERT
from transformers.models.bert.ma_bert.modeling_ma_bert import MA_BertForSequenceClassification
from transformers.models.bert.ma_bert.configuration_ma_bert import MA_BertConfig
from transformers.models.bert.modeling_bert import BertForSequenceClassification

# DistilBERT
# from transformers.models.distilbert.ma_distilbert.modeling_ma_distilbert import MA_DistilBertForSequenceClassification
# from transformers.models.distilbert.ma_distilbert.configuration_ma_distilbert import MA_DistilBertConfig
# from transformers.models.distilbert.modeling_distilbert import DistilBertForSequenceClassification

from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, load_metric, concatenate_datasets 
import datasets
import random
import argparse

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def get_student_model(num_labels, ckpt_file, model_checkpoint, teacher_ckpt_file):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Change this if using different model
    config = MA_BertConfig(use_softmax_approx = True, 
                        share_softmax_nn=True,
                        softmax_input_size = 128,
                        softmax_hidden_layer_size = 128,
                        norm_type="Power", 
                        use_linformer=False, 
                        bert_encoder_hidden_act = "relu", 
                        verbose = True, 
                        num_labels = num_labels)
    # Change this if using different model
    student_model = MA_BertForSequenceClassification.from_pretrained(model_checkpoint, 
                                                                           config = config)
    model_state_dict = student_model.state_dict()
    checkpoint = torch.load(f"{ckpt_file}", map_location=device)
    print(f"Using Checkpoint: {ckpt_file}")
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in student_model.state_dict()}
    i = 0
    for key in model_state_dict:
        if key in checkpoint:
            i+=1
            model_state_dict[key] = pretrained_dict[key]
    print(f"Done Generating State Dict from Pretrained - {i} parameters loaded")
    student_model.load_state_dict(model_state_dict)

    return student_model

def get_teacher_model(num_labels, teacher_ckpt_file, model_checkpoint):
    teacher_model = BertForSequenceClassification.from_pretrained(model_checkpoint,
                                                                  num_labels = num_labels, 
                                                                  output_hidden_states = True, # For Patient KD 
                                                                  output_attentions = True)
    teacher_ckpt = torch.load(teacher_ckpt_file, map_location=device)
    #Load finetuned BERT as teacher
    teacher_model.load_state_dict(teacher_ckpt) 
    # Freeze Teacher Parameters
    for param in teacher_model.parameters():
        param.requires_grad = False
    print("Loaded Teacher Model Successfully")
    return teacher_model

def preprocess_function_1(examples):
    for i in range(len(examples['text'])):
        examples['text'][i] = examples['text'][i].replace('<br /><br />', " ")
    return examples

def preprocess_function_2(examples):
    return tokenizer(examples['text'], max_length=128, padding = "max_length",truncation=True)

def compute_metrics(eval_preds):
    metric = load_metric("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def show_random_elements(dataset, num_examples=10):
    print(dataset)
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    for column, typ in dataset.features.items():
        if isinstance(typ, datasets.ClassLabel):
            df[column] = df[column].transform(lambda i: typ.names[i])
    pd.set_option('display.width', 120)
    print(df)

def TotalLossForKD(student_output, teacher_output, alpha, temperature, beta, num_labels):                                                                       
    soft_target_loss = nn.CrossEntropyLoss()(student_output['logits'].view(-1, num_labels)/temperature, 
                                            nn.Softmax(dim = -1)(teacher_output['logits'].view(-1, num_labels)/temperature))
    layer_loss = 0
    attn_loss = 0
    for i in range(12):
        layer_loss += nn.MSELoss()(student_output['hidden_states'][i], teacher_output['hidden_states'][i])
        attn_loss += nn.MSELoss()(student_output['attentions'][i], teacher_output['attentions'][i])
    
    layer_loss += nn.MSELoss()(student_output['hidden_states'][12], teacher_output['hidden_states'][12])
    
    return alpha * soft_target_loss \
        + (1 - alpha) * student_output.loss \
        + beta * (layer_loss + 100 * attn_loss) 

class DistillationTrainer(Trainer):
    def __init__(self, 
                *args, 
                teacher_model=None, 
                num_labels=2,
                KD_alpha=0.9, 
                **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self.num_labels = num_labels
        self.teacher.eval()
        self._move_model_to_device(self.teacher,self.model.device)
        self.KD_alpha = KD_alpha

    def compute_loss(self, model, inputs, return_outputs=False):
        student_output = model(**inputs)
        teacher_output = self.teacher(**inputs)
        loss = TotalLossForKD(student_output=student_output,
                                   teacher_output=teacher_output,
                                   alpha=self.KD_alpha,
                                   temperature=15,
                                   beta=1,
                                   num_labels=self.num_labels)
        return (loss, student_output) if return_outputs else loss

def preprocess_logits_for_metrics(logits, labels):
    return logits[0]
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--student_ckpt_file', type=str, required=True)
    parser.add_argument('--teacher_ckpt_file', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--file_name', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--KD_alpha', type=float, required=True)
    
    args = parser.parse_args()
    print(args)
    seed = random.randint(0, 1000)
    print(f"Seed: {seed}")
    print("Reminder: Check batch size, learning rate, and epoch are correct!\n")
    
    print("----Training BERT Base for IMDB Task----")
    ### Define task to fine tune bert on and to evaluate
    task = "imdb"

    ### Define what pre-trained model checkpoint you want to use 
    model_checkpoint = args.model

    ### Adjust the batch size to avoid out-of-memory errors
    batch_size = args.batch_size 

    print(f"Task Selected: {str.upper(task)}")
    
    ### Loading data required for the IMDB task
    dataset = load_dataset("imdb")

    ### Tokenizer to preprocess the input before feeding into the model
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, model_max_length = 128)

    ### Preprocess all data loaded for the IMDB task
    dataset = dataset.map(preprocess_function_1, batched=True)
    encoded_dataset = dataset.map(preprocess_function_2, batched=True)


    ### IMDB - only 2 labels - negative (0) or positive (1)
    num_labels = 2

    ### Define your model here
    ### Loading Pretrained Model
    student_model = get_student_model(num_labels, args.student_ckpt_file, model_checkpoint=model_checkpoint, teacher_ckpt_file=args.teacher_ckpt_file)
    ### Define teacher
    teacher_model = get_teacher_model(num_labels, model_checkpoint=model_checkpoint, teacher_ckpt_file=args.teacher_ckpt_file)
    

    metric_name = "accuracy"
    model_name = model_checkpoint.split("/")[-1]
    
    ###`TrainingArguments` is a class that contains all the attributes to customize 
    ### the training. It requires one folder name, which will be used to save the checkpoints 
    ### of the model, and all other arguments are optional:
    training_args = TrainingArguments(
        f"{args.save_dir}/{args.file_name}-{seed}-{task}", #Creates a directory named as provided
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        logging_strategy = "epoch",
        save_total_limit = 1,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=args.epoch,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        optim="adamw_torch",
        warmup_ratio = 0.1,
        seed=seed
    )

    trainer = DistillationTrainer(
        student_model,
        training_args,
        teacher_model=teacher_model,
        num_labels=num_labels,
        KD_alpha=args.KD_alpha,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics = preprocess_logits_for_metrics
    )
    trainer.train()

