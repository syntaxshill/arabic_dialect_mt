'''
Wrapper around the various LLMs we're trying.
Only imports the LLM if it's being used.
Can easily do LLM-specific postprocessing if necessary.
'''

class LlmTranslator:

    def __init__(self, model="gpt"):
        self.model_name = model
        
        if model == "gpt":
            from .llms import gpt
            self.request_f = gpt.request_gpt
            self.postprocess_f = gpt.postprocess

        elif model == "aya":
            from .llms import aya
            self.request_f = aya.aya_translate
            self.postprocess_f = aya.postprocess

        elif model == "gemini":
            from .llms import gemini
            self.request_f = gemini.request_gemini
            self.postprocess_f = lambda x: x
        
        else:
            raise Exception("Model not supported")


    def translate(self, src_text, src_lang, tgt_lang, return_raw=False):
        prompt = f"{src_lang}: {src_text}\n{tgt_lang}: "
        raw_text = self.request_f(prompt)
        return raw_text if return_raw else self.postprocess(raw_text)
    
    def postprocess(self, text):
        return self.postprocess_f(text)


