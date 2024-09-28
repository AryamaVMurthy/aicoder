# Desc: main entry point for the aicoder package
from feedback_loop import feedback_loop

if __name__ == "__main__":
    n = 5  # Number of iterations
    with open('problem_statement.txt', 'r') as file:
        problem_statement = file.read()
    print(feedback_loop(n, problem_statement))
    


