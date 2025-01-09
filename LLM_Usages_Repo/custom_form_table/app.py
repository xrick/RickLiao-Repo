from flask import Flask, request, jsonify, render_template
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import json
import os

app = Flask(__name__)

# 設置您的 OpenAI API 金鑰
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# 創建提示模板
form_template = PromptTemplate(
    input_variables=["description"],
    template="""基於以下描述，創建一個適合的表單結構：
    描述: {description}
    
    請生成一個 JSON 格式的表單結構，包含以下內容：
    1. 欄位名稱
    2. 欄位類型（text, number, date, select, radio, checkbox）
    3. 是否必填
    4. 驗證規則（如果需要）
    5. 欄位描述
    
    只返回 JSON 格式的結果，不要包含其他說明。"""
)

# 初始化 LangChain
llm = OpenAI(temperature=0.7)
form_chain = LLMChain(llm=llm, prompt=form_template)

# HTML 模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>自定義表單生成器</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .container { margin-top: 50px; }
        #generatedForm { margin-top: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h2>自定義表單生成器</h2>
        <div class="row">
            <div class="col-md-6">
                <div class="form-group">
                    <label for="description">描述您需要的表單：</label>
                    <textarea class="form-control" id="description" rows="4"></textarea>
                </div>
                <button class="btn btn-primary mt-3" onclick="generateForm()">生成表單</button>
            </div>
            <div class="col-md-6">
                <div id="generatedForm"></div>
            </div>
        </div>
    </div>

    <script>
        async function generateForm() {
            const description = document.getElementById('description').value;
            const response = await fetch('/generate_form', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({description: description}),
            });
            const data = await response.json();
            displayForm(data.form_structure);
        }

        function displayForm(formStructure) {
            const formDiv = document.getElementById('generatedForm');
            let formHtml = '<form class="border p-3">';
            
            formStructure.fields.forEach(field => {
                formHtml += `<div class="mb-3">`;
                
                switch(field.type) {
                    case 'text':
                    case 'number':
                    case 'date':
                        formHtml += `
                            <label class="form-label">${field.label}${field.required ? ' *' : ''}</label>
                            <input type="${field.type}" class="form-control" 
                                   name="${field.name}" ${field.required ? 'required' : ''}>`;
                        break;
                    
                    case 'select':
                        formHtml += `
                            <label class="form-label">${field.label}${field.required ? ' *' : ''}</label>
                            <select class="form-select" name="${field.name}" ${field.required ? 'required' : ''}>
                                <option value="">請選擇...</option>
                                ${field.options.map(opt => `<option value="${opt}">${opt}</option>`).join('')}
                            </select>`;
                        break;
                    
                    case 'radio':
                        formHtml += `<label class="form-label">${field.label}${field.required ? ' *' : ''}</label>`;
                        field.options.forEach(opt => {
                            formHtml += `
                                <div class="form-check">
                                    <input class="form-check-input" type="radio" name="${field.name}" value="${opt}" ${field.required ? 'required' : ''}>
                                    <label class="form-check-label">${opt}</label>
                                </div>`;
                        });
                        break;
                        
                    case 'checkbox':
                        formHtml += `<label class="form-label">${field.label}</label>`;
                        field.options.forEach(opt => {
                            formHtml += `
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" name="${field.name}" value="${opt}">
                                    <label class="form-check-label">${opt}</label>
                                </div>`;
                        });
                        break;
                }
                
                if (field.description) {
                    formHtml += `<div class="form-text">${field.description}</div>`;
                }
                
                formHtml += '</div>';
            });
            
            formHtml += '<button type="submit" class="btn btn-success">提交</button></form>';
            formDiv.innerHTML = formHtml;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    return HTML_TEMPLATE

@app.route('/generate_form', methods=['POST'])
def generate_form():
    data = request.json
    description = data.get('description', '')
    
    try:
        # 使用 LangChain 生成表單結構
        result = form_chain.run(description)
        form_structure = json.loads(result)
        
        return jsonify({
            'status': 'success',
            'form_structure': form_structure
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
