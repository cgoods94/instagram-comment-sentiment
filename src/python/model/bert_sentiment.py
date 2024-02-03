from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline


def bert_sentiment() -> pipeline:
    """Loads a BERT sentiment analysis pipeline.

    Args: None

    Returns: a transformers pipeline primed to do Bert sentiment analysis
    """
    # Load pre-trained model and tokenizer
    model_name = "bert-base-uncased"
    model = BertForSequenceClassification.from_pretrained(model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Load sentiment analysis pipeline
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    return nlp
