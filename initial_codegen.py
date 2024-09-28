from transformers import pipeline
import torch
import re
# Load the pipeline with the Qwen-Math model
def format_code_solution(input_text):
    """
    Function to format a given string containing Python code into a readable format
    with proper indentation and comments.
    """
    # Splitting the input based on newlines to format step by step
    lines = input_text.split('\n')
    
    # To keep track of the formatted output
    formatted_output = []
    
    # Loop through each line
    for line in lines:
        # Strip leading and trailing spaces
        line = line.strip()
        
        # Apply indentation based on certain keywords
        if line.startswith('def') or line.startswith('if') or line.startswith('while') or line.startswith('for'):
            formatted_output.append('\n' + line)
        elif line.startswith('import'):
            formatted_output.append('\n' + line + '\n')
        else:
            formatted_output.append('    ' + line)  # Indent regular lines
    
    # Join the formatted lines into a readable format
    return '\n'.join(formatted_output)
def understand(problem_statement):
    # Load the pipeline with the Qwen-Math model
    qwen_math_pipeline = pipeline("text-generation", model="Qwen/Qwen2.5-Math-1.5B-Instruct", torch_dtype=torch.bfloat16, device=0)
    # Generate the algorithm and logic
    prompt =[{"role": "system", "content": "You are an assistant for another coding based llm, your output algorithm and logic will be directly fed into it, You have to write the given question in mathematical and logical terms. You also have to generate the easy to understand mathematical explaination of the problem and the step by step, in detail logic without missing any information and giving information about every step that is most suited for solving the question."},
         {"role": "user", "content": f'''Here is the given problem for which the explaination and logic should be generated{problem_statement}'''},]

    result = qwen_math_pipeline(prompt, max_new_tokens=1024)
    return result
#function to write the python code using qwen 2.5 1.5b coder model, with the previous formatted output as the input

def codermodel(explaination):
    # Load the pipeline with the Qwen-Coder model
    qwen_coder_pipeline = pipeline("text-generation", model="Qwen/Qwen2.5-Coder-1.5B-Instruct", torch_dtype=torch.bfloat16, device=0)
    # Generate the Python code
    prompt = [{"role": "system", "content": "You are a competative programming model which codes in python, you should give the python code for the following and the logic and explaination on how to implement it is also given to you"},
         {"role": "user", "content": f'''Here is the generated algorithm and logic for the problem: {explaination}'''},]

    result = qwen_coder_pipeline(prompt, max_new_tokens=1024)
    return result

def format_output(text):
    # Convert escape sequences
    text = text.replace('\\n', '\n').replace('\\t', '\t')
    
    # Convert math expressions
    text = re.sub(r'\$(.*?)\$', r'$$\1$$', text)  # Convert inline math
    text = re.sub(r'\$\$(.*?)\$\$', r'$$\1$$', text)  # Convert display math

    return text

def generate_code(problem_statement):
    explaination = format_code_solution(understand(problem_statement)[0]['generated_text'][2]["content"])
    return format_output(codermodel(explaination)[0]['generated_text'][2]["content"])

if(__name__ == "__main__"):
    problem_statement = """
write a program to transpose a 2d array square matrix using recursion
input: n = number of rows;
n elements of the array;
"""
    explaination = format_code_solution(understand(problem_statement)[0]['generated_text'][2]["content"])
    print(format_output(codermodel(explaination)[0]['generated_text'][2]["content"]))