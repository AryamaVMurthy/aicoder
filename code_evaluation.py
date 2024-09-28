from transformers import pipeline
from initial_codegen import generate_code
import torch
import re
def get_test_cases(problem_statement):
    # Initialize qwen for test case generation
    model = pipeline("text-generation", model="Qwen/Qwen2.5-Math-1.5B-Instruct",torch_dtype=torch.bfloat16, device=0)

    # Generate test cases based on the problem statement
    prompt = [{"role": "system", "content": """you are an assistant that generates test cases for the given problem statement, you have to generate 10 test cases based on the problem statement given below
               Give the output in the exact format of [[[testcase1,testcase2,testcase3....testcase10]]]
    example for generating integers as input test case: [[[1,2,3,4,5,6,7,8,9,10]]]"""},
         {"role": "user", "content":f"""
    Generate 10 random test cases based on the following problem statement:
    
    Problem Statement:
    {problem_statement}
    
    Ensure the test cases cover a wide range of input values and edge cases but follows the constraints given in the question if any.
    IMPORTANT: GIVE THE OUTPUT IN THE EXACT FORMAT OF <[[[testcase1,testcase2,testcase3,....,testcase10]]]>
    DO NOT GENERATE PYTHON CODE FOR TESTS, JUST GIVE THE TEST CASES DIRECTLY IN THE ABOVE FORMAT.
    """ },]
  
    test_cases = model(prompt, max_new_tokens=1024)
    print(test_cases)
    match =  re.findall(r'[[(.*?)]]', test_cases[0]['generated_text'][2]["content"])[0]
    return eval(match)

def feedback(code, problem_statement):
    # Initialize StarCoder for syntax check and test case evaluation
    model = pipeline("text-generation", model="Qwen/Qwen2.5-Math-1.5B-Instruct")

    # Check syntax and run test cases

    prompt = [{"role": "system", "content": f""" You are an AI assistant designed to evaluate and provide feedback on Python code for a competative programming problem. Then give the logic of the corrected version of the code. Here's your task:

Check Syntax:

Evaluate the given Python code for any syntax errors.
If syntax is incorrect, mark all test cases as failed.
Execute Test Case:

If the syntax is correct, execute the code with generating 10 test cases for it.
Verify the correctness of the output based on the expected behavior.
Feedback on Test Case Results:

For each test case, provide results in the format pass or fail.
If any test fails, provide feedback on what went wrong and suggest improvements for syntax errors, logical errors, or output mismatches.
Final Output:
    
Provide the exact feedback and summary of code evaluation in the form of Feedback:
If all test cases pass without errors and the code logic is correct, output <[[[[PASSCOMPLETE]]]]>.
Otherwise, output test results as [result1, result2, result3... result10], where each result is 'pass' or 'fail' string 
                   Based on the above test results, provide the logic of the corrected version of the code in the form of Logic:
                   """}, {"role": "user", "content": f"""Code:
        {code}


Problem Statement {problem_statement}"""},]
    passed = False
        
    response = model(prompt, max_new_tokens=1024)
    if "PASSCOMPLETE" in response[0]['generated_text'][2]["content"]:
        passed = True
        return passed,""
    else:
        returned_feedback = response[0]['generated_text'][2]["content"]
        

    
    return passed, returned_feedback


    

# Example usage
if __name__ == "__main__":
    with open('problem_statement.txt', 'r') as file:
        problem_statement = file.read()
    code = generate_code(problem_statement)
    print(feedback(code,problem_statement))
    