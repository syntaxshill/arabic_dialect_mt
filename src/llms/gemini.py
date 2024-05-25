import os
import google.generativeai as genai


GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-pro')

def request_gemini(prompt):
    response = model.generate_content(prompt)
    try:
        return response.text
    except:
        print("Content filtered: ", prompt)
        print(response.candidates[0].finish_reason)
        print(response.candidates[0].safety_ratings)
        return "Content filtered"

# doesn't seem to need postprocessing!