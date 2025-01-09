from flask import Flask, render_template, request, jsonify
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.callbacks import get_openai_callback
import os

app = Flask(__name__)

# 設置 OpenAI API 金鑰
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# 創建 LLM 實例
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

# 創建記憶體實例
memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history"
)

# 定義數學教學助手的系統提示詞
MATH_TEACHER_TEMPLATE = """你是一位經驗豐富的國小數學教學顧問。你的任務是協助教師解決數學教學相關的問題。

請注意以下幾點：
1. 回答要符合國小學生的認知水平
2. 提供具體的教學建議和例子
3. 使用適當的視覺化建議
4. 考慮不同學習程度的學生需求
5. 提供多元的教學策略

聊天歷史：
{chat_history}

教師問題：{question}

請提供專業且實用的建議："""

# 創建提示詞模板
prompt = ChatPromptTemplate.from_template(MATH_TEACHER_TEMPLATE)

# 創建對話鏈
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        question = request.json['question']
        
        # 使用 LangChain 處理問題
        with get_openai_callback() as cb:
            response = conversation.predict(question=question)
            tokens_used = cb.total_tokens
        
        return jsonify({
            'answer': response,
            'tokens': tokens_used
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)