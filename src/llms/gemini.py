import os
import google.generativeai as genai


GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-pro')

def request_gemini(prompt):
    response = model.generate_content(prompt)
    return response.text

# doesn't seem to need postprocessing!