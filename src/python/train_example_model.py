from training import gather_and_clean_training_set
from model import train_bert_sentiment

training_fp = "../../data/training_comments_done.xlsx"
tokenizer_save_path = "model/tokenizers/"

training_dataset = gather_and_clean_training_set(training_fp, tokenizer_save_path)

# This saves the model and tokenizer as a side effects and returns
# the loss values to you for data viz/monitoring.
loss_values = train_bert_sentiment(
    training_dataset, train_size=0.8, epochs=4, device_type="cpu"
)
