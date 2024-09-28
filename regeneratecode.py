from code_evaluation import evaluate_code
from initial_codegen import generate_code
from initial_codegen import format_output
from code_evaluation import feedback
import torch
from transformers import pipeline
def regenerate_code(code, feedback):
    # Load the pipeline with the Qwen-Coder model
    qwen_coder_pipeline = pipeline("text-generation", model="Qwen/Qwen2.5-Coder-1.5B-Instruct", torch_dtype=torch.bfloat16, device=0)
    # Generate the Python code
    prompt = [{"role": "system", "content": "You are a competative programming model which codes in python, you should give the python code that is improved from its previous code for the following. the previously written code and feedback is given to you"},
         {"role": "user", "content": f'''
          Here is the previous code:    
          {code}
          
          Here is the feedback and the corrected logic for the code:
          {feedback}
            
'''},]

    result = qwen_coder_pipeline(prompt, max_new_tokens=1024)
    return format_output(result)
       

# Example usage
if(__name__ == "__main__"):
    with open('problem_statement.txt', 'r') as file:
        problem_statement = file.read()
    code = generate_code(problem_statement)
    passed, returned_feedback = feedback(problem_statement, code)
    
