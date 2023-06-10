import tensorflow as tf
import keras 
from embeddings import embedding

class nlp_to_sql(tf.keras.Model):
    def __init__(self,nlp_text_processor,sql_text_processor,fixed_embedding,unit=128):
        super().__init__()
        #natural language
        self.nlp_text_processor=nlp_text_processor
        self.nlp_voba_size = len(nlp_text_processor.get_vocabulary())
        self.nlp_embedding = tf.keras.layers.Embedding(
            self.nlp_voba_size,
            output_dim=unit,
            mask_zero=True
        )
        self.fixed_embedding = fixed_embedding
        self.nlp_rnn=tf.keras.layers.Bidirectional(layers=tf.keras.layers.LSTM(int(unit/2),return_sequences=True, return_state=True))

        #attention
        self.attention = tf.keras.layers.Attention()

        #SQL
        self.sql_text_processor=sql_text_processor
        self.sql_voba_size=len(sql_text_processor.get_vocabulary())
        self.sql_embedding = tf.keras.layers.Embedding(
            self.sql_voba_size,
            output_dim=unit,
            mask_zero=True
        )
        self.sql_rnn=tf.keras.layers.LSTM(unit,return_sequences=True,return_state=True)
         
         #output
        self.out = tf.keras.layers.Dense(self.sql_voba_size)

    def call(self,nlp_text,sql_text,training=True):
        nlp_tokens = self.nlp_text_processor(nlp_text)
        nlp_vectors= self.nlp_embedding(nlp_tokens, training=training)
        nlp_fixed_vectors=self.fixed_embedding(nlp_tokens)
        nlp_combined_vectors = tf.concat([nlp_vectors,nlp_fixed_vectors],-1)
        nlp_rnn_out, fhstate, fcstate, bhstate, bcstate=self.nlp_rnn(nlp_vectors,training=training)
        nlp_hstate = tf.concat([fhstate,bhstate],-1)
        nlp_cstate=tf.concat([fcstate,bcstate],-1)

        sql_tokens = self.sql_text_processor(sql_text)
        expected = sql_tokens[:,:1:]

        teacher_forcing = sql_tokens[:,:-1]
        sql_vector=self.sql_embedding(teacher_forcing,training=training)
        sql_in = self.attention(input=[sql_vector,nlp_rnn_out], mask=[sql_vector._keras_mask],training=training)

        trans_vectors, _, _= self.sql_rnn(sql_in,initial_state=[nlp_hstate,nlp_cstate],training=training)
        out=self.out(trans_vectors,training=training)
        return out, expected, out.keras_mask


