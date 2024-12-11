import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer

class CombinedEmbeddingModel(tf.keras.Model):
    def __init__(self, num_secondary_embeddings, embedding_dim, dropout_rate=0.3):
        """
        A model that uses BERT for token encodings, concatenates with secondary embeddings,
        and applies a fully connected neural network for binary classification.
        """
        super(CombinedEmbeddingModel, self).__init__()
        self.bert = TFAutoModel.from_pretrained("bert-base-uncased", trainable=False)

        self.secondary_embedding = tf.keras.layers.Embedding(input_dim=num_secondary_embeddings, output_dim=embedding_dim)
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        # Fully connected network
        self.fc1 = tf.keras.layers.Dense(256, activation='relu')
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        self.out_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        """
        Forward pass through the model.

        Args:
        - inputs (tuple): A tuple containing (token_ids, attention_mask, secondary_indices).
          - token_ids: Tokenized inputs for BERT (shape: batch_size x seq_length).
          - attention_mask: Attention masks for BERT (shape: batch_size x seq_length).
          - secondary_indices: Indices for secondary embeddings (shape: batch_size x seq_length).

        Returns:
        - logits: Final sigmoid-activated outputs (shape: batch_size x 1).
        """
        token_ids, attention_mask, secondary_indices = inputs

        # BERT encodings
        bert_outputs = self.bert(input_ids=token_ids, attention_mask=attention_mask)
        bert_encodings = bert_outputs.last_hidden_state  # shape: (batch_size x seq_length x hidden_dim)

        # Secondary embeddings
        secondary_encodings = self.secondary_embedding(secondary_indices)  # shape: (batch_size x seq_length x embedding_dim)
        secondary_encodings = self.dropout2(secondary_encodings)
        # Concatenate along the last dimension
        combined_encodings = tf.concat([bert_encodings, secondary_encodings], axis=-1)
        # mean to pool down to lower dimension
        pooled_encodings = tf.reduce_mean(combined_encodings, axis=1)  # shape: (batch_size x (hidden_dim + embedding_dim))

        # Fully connected network
        x = self.fc1(pooled_encodings)
        x = self.dropout(x)
        x = self.fc2(x)
        logits = self.out_layer(x)

        return logits
