import os
import openai
import streamlit as st
from streamlit_chat import message

openai_api_key = os.environ.get("OPENAI_API_KEY", "")
assert openai_api_key, "Please set the OPENAI_API_KEY environment variable"
openai.api_key = openai_api_key


def generate_response(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message["content"]


# We will get the user's input by calling the get_text function
def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text


# Creating the chatbot interface
st.title("chatBot : Streamlit + openAI")

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'context' not in st.session_state:
    st.session_state['context'] = [{'role': 'system', 'content': """
You are OrderBot, an automated service to collect orders for an apparel shop. \
You first greet the customer, then collects the order, \
and then asks if it's a pickup or delivery. \
You wait to collect the entire order, then summarize it and check for a final \
time if the customer wants to add anything else. \
If it's a delivery, you ask for an address. \
Finally you collect the payment.\
You respond in a short, very conversational friendly style. \
Make sure to clarify all options, extras and sizes to uniquely \
you can get orders for :
T_shirts 10.00 GBP
pants 20 pounds
wallets 5 GBP
hand bag 15 pounds
wallet 4
sunglasses 100
"""}]

user_input = get_text()

if user_input:
    st.session_state.context.append({'role': 'user', 'content': f"{user_input}"})
    # in order for the context not to grow above the limit(4000 tokens), I'm limiting the context to the first 3 (to
    # set the tone) and the last 20 items.
    response = generate_response(st.session_state.context[:3] + st.session_state.context[-20:])
    st.session_state.context.append({'role': 'assistant', 'content': f"{response}"})

    # store the output
    st.session_state.past.append(user_input)
    st.session_state.generated.append(response)


if st.session_state['generated']:

    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
