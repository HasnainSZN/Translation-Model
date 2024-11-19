import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.models import Model

class LanguageTranslationModel:
    def __init__(self, source_texts, target_texts, max_seq_length=20, latent_dim=256):
        """
        Initialize the sequence-to-sequence translation model with verbose debugging.
        """
        # Preprocessing
        self.source_texts = ['start ' + text + ' end' for text in source_texts]
        self.target_texts = ['start ' + text + ' end' for text in target_texts]
        
        # Tokenization
        self.source_tokenizer = Tokenizer(filters='', lower=True)
        self.source_tokenizer.fit_on_texts(self.source_texts)
        source_sequences = self.source_tokenizer.texts_to_sequences(self.source_texts)
        
        self.target_tokenizer = Tokenizer(filters='', lower=True)
        self.target_tokenizer.fit_on_texts(self.target_texts)
        target_sequences = self.target_tokenizer.texts_to_sequences(self.target_texts)
        
        # Print vocabularies for debugging
        print("Source Vocabulary:", self.source_tokenizer.word_index)
        print("Target Vocabulary:", self.target_tokenizer.word_index)
        
        # Padding
        self.max_seq_length = max_seq_length
        self.encoder_input_data = pad_sequences(source_sequences, maxlen=max_seq_length, padding='post')
        self.decoder_input_data = pad_sequences(target_sequences, maxlen=max_seq_length, padding='post')
        
        # Target data preparation
        self.decoder_target_data = np.zeros_like(self.decoder_input_data)
        self.decoder_target_data[:, :-1] = self.decoder_input_data[:, 1:]
        
        # Vocabulary sizes
        self.num_encoder_tokens = len(self.source_tokenizer.word_index) + 1
        self.num_decoder_tokens = len(self.target_tokenizer.word_index) + 1
        
        self.latent_dim = latent_dim
        
        # Inverse word index for easy lookup
        self.source_reverse_word_index = {v: k for k, v in self.source_tokenizer.word_index.items()}
        self.target_reverse_word_index = {v: k for k, v in self.target_tokenizer.word_index.items()}
        
    def build_model(self):
        """
        Build sequence-to-sequence model with detailed architecture
        """
        # Encoder
        encoder_inputs = Input(shape=(self.max_seq_length,))
        encoder_embedding = Embedding(self.num_encoder_tokens, self.latent_dim, mask_zero=True)(encoder_inputs)
        encoder = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_embedding)
        encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_inputs = Input(shape=(self.max_seq_length,))
        decoder_embedding = Embedding(self.num_decoder_tokens, self.latent_dim, mask_zero=True)(decoder_inputs)
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        
        # Output layer
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        # Compile model
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Print model summary for debugging
        self.model.summary()
        
    def train(self, epochs=50, batch_size=32):
        """
        Train the translation model with verbose output
        """
        history = self.model.fit(
            [self.encoder_input_data, self.decoder_input_data], 
            np.expand_dims(self.decoder_target_data, -1),
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            verbose=1
        )
        return history
    
    def translate(self, source_sentence):
        """
        Translate a source sentence with comprehensive debugging
        """
        # Prepare input sequence
        input_text = 'start ' + source_sentence + ' end'
        print(f"Input text for translation: {input_text}")
        
        # Tokenize input
        input_seq = self.source_tokenizer.texts_to_sequences([input_text])
        input_seq = pad_sequences(input_seq, maxlen=self.max_seq_length, padding='post')
        
        # Debug input sequence
        print("Input Sequence:", input_seq)
        print("Input Tokens:", [self.source_reverse_word_index.get(idx, '<UNK>') for idx in input_seq[0] if idx != 0])
        
        # Initialize decoder input
        start_token = self.target_tokenizer.word_index['start']
        target_seq = np.zeros((1, self.max_seq_length))
        target_seq[0, 0] = start_token
        
        # Translation process
        translated_words = []
        for step in range(self.max_seq_length):
            # Predict next tokens
            prediction = self.model.predict([input_seq, target_seq])
            
            # Get the most probable token
            sampled_token_index = np.argmax(prediction[0, step, :])
            sampled_word = self.target_reverse_word_index.get(sampled_token_index, '')
            
            # Debug translation steps
            print(f"Step {step}: Token Index {sampled_token_index}, Word: {sampled_word}")
            
            # Stop conditions
            if sampled_word in ['end', ''] or len(translated_words) > self.max_seq_length:
                break
            
            if sampled_word not in ['start', 'end']:
                translated_words.append(sampled_word)
            
            # Update target sequence
            target_seq[0, step+1] = sampled_token_index
        
        return ' '.join(translated_words)

# Example usage
def main():
    # Expanded training data
    source_texts = [
        "hello how are you", 
        "what is your name", 
        "where are you from",
        "i am doing well",
        "nice to meet you"
    ]
    target_texts = [
        "bonjour comment allez-vous", 
        "quel est votre nom", 
        "d'où venez-vous",
        "je vais bien",
        "ravi de vous rencontrer"
    ]
    
    # Create and train translation model
    translation_model = LanguageTranslationModel(source_texts, target_texts)
    translation_model.build_model()
    history = translation_model.train(epochs=500)
    
    # Test translation with multiple sentences
    test_sentences = [
        "hello how are you",
        "what is your name", 
        "where are you from"
    ]
    
    for sentence in test_sentences:
        print("\n--- Translation Test ---")
        translated = translation_model.translate(sentence)
        print(f"Source: {sentence}")
        print(f"Translated: {translated}")

if __name__ == "__main__":
    main()