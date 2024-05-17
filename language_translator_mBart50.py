from transformers import  MBartForConditionalGeneration, MBart50Tokenizer
import streamlit as st

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def download_model():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBart50Tokenizer.from_pretrained(model_name)
    return model,tokenizer

st.title('Tamil to English Translator')
text = st.text_area("Enter text:",value='',height=None,max_chars=None,key=None)
model,tokenizer = download_model()

if st.button('Translate to English'):
    if text == '':
        st.write("Please enter tamil text for translation")
    else:
        model_name = "facebook/mbart-large-50-many-to-many-mmt"
        tokenizer.src_lang = "ta_IN"   
        encoded_tamil_text = tokenizer(text,return_tensors="pt")
        generated_tokens = model.generate(**encoded_tamil_text, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
        out = tokenizer.batch_decode(generated_tokens,skip_special_tokens=True)
        st.write('',str(out).strip('][\''))
else:
    pass    