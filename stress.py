import argparse

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from tqdm.auto import tqdm

import utils

llm = ChatOpenAI(model_name='gpt-4o', temperature=0)

gen_and_slow_code = PromptTemplate.from_template("""
You are given a competitive programming problem statement.
Your task is to stress test a fast solution for the given problem.
The fast solution is already written and compiled.
You need to write a slow solution.

When writing the slow solution, you should focus on correctness rather than efficiency. For example, if the problem requires you to calculate a 1D DP, you can use brute force to calculate the answer.
Before writing the slow solution, decompose the task into smaller subtasks. Lay out the steps you need to take to write the slow solution. Try to understand the complexity of the slow solution.

When writing the test case generator, you should generate random test cases for the given problem. The test cases should be small and random.
The generator should output the test case to the standard output. It should be written in C++. It shouldn't take any input.
Generate test cases that are small enough to run the slow solution in about 100ms.

Think before you write code. Decompose the task into smaller subtasks. Lay out the steps you need to take to write the slow solution and the generator.
After writing the slow solution and the generator, they will stress test the fast solution.

First, write the slow solution. The slow solution should be written in ```cpp``` code block.
Second, write the generator. The generator should be written in ```cpp``` code block.

Problem Statement:
{problem_statement}
""")


def stress_test(input_src: str, codeforces_url: str, compiler_command: str, fast_path="./fast",
                slow_code_path="./slow.cpp", slow_path="./slow",
                gen_code_path="./gen.cpp", gen_path="./gen", iters=1000):
    # Get the problem statement from the given Codeforces URL
    statement = utils.get_codeforces_statement(codeforces_url)

    # Compile the slow solution
    utils.compile(compiler_command, input_src, fast_path)

    # Slow solution and generator code from LLM
    slow_code, generator_code = (
            gen_and_slow_code | llm | RunnableLambda(utils.extract_code)
    ).invoke(input={'problem_statement': statement})

    # Write the slow solution and generator code to files
    with open(slow_code_path, 'w') as f:
        f.write(slow_code)
    with open(gen_code_path, 'w') as f:
        f.write(generator_code)

    # Compile the slow solution and generator
    utils.compile(compiler_command, slow_code_path, slow_path)
    utils.compile(compiler_command, gen_code_path, gen_path)

    # Stress test
    for _ in tqdm(range(iters), desc="Stress Testing", total=iters):
        # Generate test case
        gen_output, _ = utils.invoke(gen_path, '')
        # Run the slow solution
        slow_output, _ = utils.invoke(slow_path, gen_output)
        # Run the fast solution
        fast_output, _ = utils.invoke(fast_path, gen_output)

        if slow_output.strip() != fast_output.strip():
            print(f"Test case:\n {gen_output}")
            print(f"Slow output:\n {slow_output}")
            print(f"Fast output:\n {fast_output}")
            print("Outputs do not match.")
            return
    print("All outputs match. Stress test passed.")


def main():
    parser = argparse.ArgumentParser(description='Demo Auto Stress testing.')
    parser.add_argument('input_src', type=str, help='Path to the input source file')
    parser.add_argument('codeforces_url', type=str, help='Codeforces problem URL')
    parser.add_argument('--compiler_command', type=str, help='Optional compiler command', default='g++-14')

    args = parser.parse_args()
    print(args)
    stress_test(args.input_src, args.codeforces_url, args.compiler_command)


if __name__ == '__main__':
    main()
