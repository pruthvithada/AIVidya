from flask import Flask, request, render_template_string, redirect, url_for, flash
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
import PyPDF2
import os
import io

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure key

# Load environment variables from .env file
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

HTML_FORM = '''
<!doctype html>
<title>Chat Your PDFs (Flask)</title>
<h2>Chat Your PDFs</h2>
<form method=post enctype=multipart/form-data>
  <label>Upload a PDF file:</label><br>
  <input type=file name=pdf_file accept="application/pdf"><br><br>
  <label>Ask a Question:</label><br>
  <input type=text name=question style="width:400px"><br><br>
  <input type=submit value="Get Answer">
</form>
{% with messages = get_flashed_messages() %}
  {% if messages %}
    <ul style="color:red;">{% for message in messages %}<li>{{ message }}</li>{% endfor %}</ul>
  {% endif %}
{% endwith %}
{% if answer %}
  <h3>Answer:</h3>
  <div style="white-space: pre-wrap;">{{ answer }}</div>
{% endif %}
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    if request.method == 'POST':
        if not google_api_key:
            flash("API key not found. Please set the GOOGLE_API_KEY environment variable.")
            return render_template_string(HTML_FORM, answer=None)
        pdf_file = request.files.get('pdf_file')
        user_question = request.form.get('question', '').strip()
        if not pdf_file or pdf_file.filename == '':
            flash('Please upload a PDF file.')
            return render_template_string(HTML_FORM, answer=None)
        if not user_question:
            flash('Please enter a question.')
            return render_template_string(HTML_FORM, answer=None)
        try:
            pdf_data = pdf_file.read()
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
            pdf_pages = pdf_reader.pages
            context = "\n\n".join(page.extract_text() for page in pdf_pages)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
            texts = text_splitter.split_text(context)
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_index = Chroma.from_texts(
                texts,
                embeddings,
                persist_directory="./chroma_db"
            ).as_retriever()
            docs = vector_index.get_relevant_documents(user_question)
            prompt_template = """
            Answer the question as detailed as possible from the provided context,
            make sure to provide all the details, if the answer is not in
            provided context just say, "answer is not available in the context",
            don't provide the wrong answer\n\n
            Context:\n {context}?\n
            Question: \n{question}\n
            Answer:
            """
            prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
            model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, api_key=google_api_key)
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            answer = response.get('output_text', 'No answer returned.')
        except Exception as e:
            flash(f"Error: {str(e)}")
            return render_template_string(HTML_FORM, answer=None)
    return render_template_string(HTML_FORM, answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
