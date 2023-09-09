from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from transformers import AutoTokenizer
base_model_name = "meta-llama/Llama-2-13b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    print("logits: ")
    print(logits)
    print("labels: ")
    print(labels)
    return None

def preprocess_logits_for_metrics(logits, labels):
    print("logits: ")
    # print(logits[0])
    print(logits[0].shape)
    print(tokenizer.batch_decode(logits[0], skip_special_tokens=True))
    print("labels: ")
    # print(labels)
    print(labels.shape)
    print(tokenizer.batch_decode(labels, skip_special_tokens=True))
    assert 1==2