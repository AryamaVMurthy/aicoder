from code_evaluation import feedback
from initial_codegen import generate_code
from regeneratecode import regenerate_code



def feedback_loop(iterations,problem_statement):
    code = generate_code(problem_statement)
    for i in range(0, iterations-1):
        passed, returned_feedback = feedback(problem_statement,code)
        if(passed):
            return code
        else:
            code = regenerate_code(code, returned_feedback)
    return code
if __name__ == "__main__":
    n = 5  # Number of iterations
    with open('problem_statement.txt', 'r') as file:
        problem_statement = file.read()   
    feedback_loop(n, problem_statement)