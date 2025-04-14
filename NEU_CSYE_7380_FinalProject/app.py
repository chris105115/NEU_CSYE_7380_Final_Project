import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime
import torch
import gc
import faiss
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from huggingface_hub import login

# ========================== Section 1: Hugging Face Login ==========================
# Login to Hugging Face to access gated models (required for some models)
login("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")

# ========================== Section 2: GPU Setup ==========================
# Check and release GPU memory
for obj in gc.get_objects():
    try:
        if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
            pass  # Optionally print object info for debugging
    except:
        pass

torch.cuda.empty_cache()

# ========================== Section 3: Model Initialization ==========================
# Change embedding model here
embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
# embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Quantization config for memory-efficient LLM loading
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Change LLM model here
model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name = "mistralai/Mistral-7B-Instruct-v0.1"

# Load tokenizer and model with quantization
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.to("cuda")

# ========================== Section 4: FAISS Search ==========================
class ResearchAssistant_FAISS:
    def __init__(self, csv_path):
        self.embedding_model = embedding_model
        try:
            self.data = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                self.data = pd.read_csv(csv_path, encoding='windows-1252')
            except UnicodeDecodeError:
                self.data = pd.read_csv(csv_path, encoding='latin-1')

        self.questions = self.data["Question"].tolist()
        self.answers = self.data["Answer"].tolist()
        self._build_index()

    def _build_index(self):
        self.embeddings = self.embedding_model.encode(self.answers, convert_to_numpy=True)
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)

    def answer_question(self, question, k=10):
        q_embedding = self.embedding_model.encode([question])[0].reshape(1, -1)
        distances, indices = self.index.search(q_embedding, k)
        return [self.answers[idx] for idx in indices[0]]

# ========================== Section 5: RAG Class ==========================
class ResearchAssistant_FAISS_With_Model:
    def __init__(self):
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.model = model
        self.index = None

    def generate_answer_from_csv(self, question: str, csv_path: str, k: int = 5) -> str:
        assistant = ResearchAssistant_FAISS(csv_path)
        retrieved_answers = assistant.answer_question(question, k)
        context = "\n".join(retrieved_answers)

        prompt = f"""Given the following information, answer the question:

        The combination of k answers from the CSV file:
        {context}

        Question:
        {question}

        Answer:
        """

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.7,
                num_return_sequences=1,
                do_sample=True
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()

# ========================== app.py ==========================

class StockTradingEnv:
    def __init__(self, stock_prices):
        self.stock_prices = stock_prices
        self.current_step = 0
        self.max_steps = len(stock_prices)
        self.position = 0
        self.cash = 50000
        self.stock_owned = 100
        self.total_reward = self.cash
        self.portfolio_value = self.cash

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.cash = 50000 - 100 * self.stock_prices[0]
        self.stock_owned = 100
        self.total_reward = 0
        self.portfolio_value = self.cash
        return self._get_state()

    def _get_state(self):
        return np.array([self.stock_prices[self.current_step], self.position])

    def step(self, action):
        current_price = self.stock_prices[self.current_step]
        reward = self.portfolio_value

        if action == 1 and self.position == 0:
            self.stock_owned = self.cash // current_price
            self.cash = self.portfolio_value - self.stock_owned * current_price
            reward = self.cash + self.stock_owned * current_price
            self.position = 1
        elif action == 2 and self.position == 1:
            self.cash = self.cash + self.stock_owned * current_price
            reward = self.cash
            self.stock_owned = 0
            self.position = 0

        self.portfolio_value = self.cash + self.stock_owned * current_price

        self.current_step += 1
        done = self.current_step >= self.max_steps - 1
        next_state = self._get_state() if not done else np.array([current_price, self.position])
        return next_state, reward, done

    def get_possible_actions(self):
        return [0, 1, 2]

def discretize(price, stock_prices, n_bins=100):
    min_price, max_price = min(stock_prices), max(stock_prices)
    bin_width = (max_price - min_price) / n_bins
    return min(n_bins - 1, max(0, int((price - min_price) // bin_width)))

def q_learning(env, stock_prices, q_table, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            price_bin = discretize(state[0], stock_prices)
            position = int(state[1])
            if random.uniform(0, 1) < epsilon:
                action = random.choice(env.get_possible_actions())
            else:
                action = np.argmax(q_table[price_bin, position])
            next_state, reward, done = env.step(action)
            next_price_bin = discretize(next_state[0], stock_prices)
            next_position = int(next_state[1])
            best_next_action = np.argmax(q_table[next_price_bin, next_position])
            q_table[price_bin, position, action] = q_table[price_bin, position, action] + alpha * (
                reward + gamma * q_table[next_price_bin, next_position, best_next_action] - q_table[price_bin, position, action])
            state = next_state
    return q_table

def test_policy(env, stock_prices, q_table, dates_vec):
    state = env.reset()
    done = False
    steps = 0
    log_messages = []
    portfolio_values = []
    while not done:
        price_bin = discretize(state[0], stock_prices)
        position = int(state[1])
        action = np.argmax(q_table[price_bin, position])
        state, reward, done = env.step(action)
        steps += 1
        portfolio_values.append(env.portfolio_value)
        log_line = f"{dates_vec[steps]}: Action {['Hold', 'Buy', 'Sell'][action]}, Stock Price: ${state[0]:.2f}, Portfolio Value: {env.portfolio_value:.2f}"
        log_messages.append(log_line)
    pnl_df = pd.DataFrame({'Portfolio Value': portfolio_values}, index=range(1, steps+1))
    return pnl_df, log_messages

def wyckoff_chatbot(user_input):
    csv_path = "Team_Wykoff_QA.csv"
    RAG_with_FAISS = ResearchAssistant_FAISS_With_Model()
    response = RAG_with_FAISS.generate_answer_from_csv(user_input, csv_path)
    return response

st.title("Richard Wyckoff Trading Insights & Back-testing Dashboard")

mode = st.sidebar.radio("Navigation", ["Chatbot", "Back-testing Dashboard"])

if mode == "Chatbot":
    st.header("Chat with Richard Wyckoff")
    user_question = st.text_input("Ask a question about investment insights:")
    if st.button("Send"):
        response = wyckoff_chatbot(user_question)
        st.write("**Wyckoff's Insight:**", response)

if mode == "Back-testing Dashboard":
    st.header("Back-testing Trading Strategy")
    st.markdown("### Configure Back-test Parameters")
    stock_symbol = st.text_input("Stock Ticker", value="NVDA")
    start_date = st.date_input("Start Date", value=datetime(2024, 1, 1))
    end_date = st.date_input("End Date", value=datetime(2024, 12, 31))
    episodes = st.number_input("Number of Training Episodes", min_value=100, max_value=5000, value=1000, step=100)

    if st.button("Run Back-test"):
        data = yf.download(stock_symbol, start=start_date, end=end_date)
        if data.empty:
            st.error("No data found for the selected parameters. Please try again.")
        else:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    stock_data = data.xs(stock_symbol, axis=1, level='Ticker')
                else:
                    stock_data = data
                stock_prices = stock_data['Close'].values
            except Exception as e:
                st.error("Error processing stock data: " + str(e))
                stock_prices = data['Close'].values

            dates_vec = stock_data.index.date.astype(str)

            env = StockTradingEnv(stock_prices)
            q_table = np.zeros((100, 2, 3))
            with st.spinner("Training the model..."):
                q_table = q_learning(env, stock_prices, q_table, episodes=episodes)
            pnl_df, log_messages = test_policy(env, stock_prices, q_table, dates_vec)
            st.success("Back-test completed.")

            st.markdown("### PnL Dynamics Chart")
            st.line_chart(pnl_df["Portfolio Value"])

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(pnl_df.index, pnl_df["Portfolio Value"])
            ax.set_title(f"{stock_symbol} - Strategy PnL Dynamics")
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Portfolio Value")
            ax.grid(True)
            st.pyplot(fig)

            st.markdown("### Trading Log Data")
            st.text_area("Logs", "\n".join(log_messages), height=300)