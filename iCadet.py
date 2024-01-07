import streamlit as st
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def main():
    st.image('mmu_logo.png', width=300)  # Replace 'your_logo.png' with the path to your logo image
    st.header('iCadet Programme')
    
    df = pd.read_csv('iCadet_embedding.csv')

    prompt = st.text_input('Ask a question:')
    
    if st.button('Get Answer'):
        if prompt:

            def createEmbedding(row):
                return model.encode(row['Question'], convert_to_tensor=True)
            
            def calcDist(row):
                return np.array(util.pytorch_cos_sim(row['Q_embedding'], prompt_embedding))[0][0]

            prompt_embedding = model.encode(prompt, convert_to_tensor=True)
            df['Q_embedding'] = df.apply(createEmbedding, axis=1)
            df['Similarity'] = df.apply(calcDist, axis=1)
            df = df.sort_values(by='Similarity', ascending=False).reset_index()
            if df.loc[0,'Similarity']<0.7:
                st.warning('Please try another question!')
            else:
                st.success(df.loc[0,'Answer'])
        else:
            st.warning('Please enter a question.')

    


if __name__ == '__main__':
    main()