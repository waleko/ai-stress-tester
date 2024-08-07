import argparse

from langchain import hub
from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from tqdm.auto import tqdm

import utils

llm = ChatOpenAI(model_name='gpt-4o', temperature=0)

# Instructions for the agent to stress test a fast solution
agent_prompt = PromptTemplate.from_template("""
You are given a competitive programming problem statement.
Your task is to stress test a fast solution for the given problem.
The fast solution is already written and compiled.
You need to write a slow solution.

When writing the slow solution, you should focus on correctness rather than efficiency. For example, if the problem requires you to calculate a 1D DP, you can use brute force to calculate the answer.
Before writing the slow solution, decompose the task into smaller subtasks. Lay out the steps you need to take to write the slow solution. Try to understand the complexity of the slow solution.

When writing the test case generator, you should generate random test cases for the given problem. The test cases should be small and random.
The generator should output the test case to the standard output. It should be written in C++. It shouldn't take any input.
Generate test cases that are small enough to run the slow solution in about 100ms.

Think before you write code into files. Decompose the task into smaller subtasks. Lay out the steps you need to take to write the slow solution and the generator.
After writing the slow solution and the generator, you should stress test the fast solution using the slow solution and the generator.

Slow solution should be written in slow.cpp and generator should be written in gen.cpp.

Don't try using multiple tools at once. Think about the slow solution first, then the generator, and finally the stress test.

Problem Statement:
{problem_statement}

Don't stop until you have run the stress test.

Don't run multiple tools at once. First, write the slow solution, then the generator, and finally the stress test.

At the end, deliver your verdict. Is the fast solution correct?
""")

# Instructions for the LLM to output everything in JSON format
chat_prompt = hub.pull("hwchase17/react-chat-json")


def stress_test(input_src: str, codeforces_url: str, compiler_command=None, fast_path="./fast",
                slow_code_path="./slow.cpp", slow_path="./slow",
                gen_code_path="./gen.cpp", gen_path="./gen", iters=1000):
    # Get the problem statement from the given Codeforces URL
    statement = utils.get_codeforces_statement(codeforces_url)

    # Compile the slow solution
    utils.compile(compiler_command, input_src, fast_path)

    @tool
    def write_to_file(file_path: str, content: str) -> None:
        """
        Write the content to the given file.
        """
        with open(file_path, 'w') as f:
            f.write(content)

    @tool
    def run_stress(slow_code_path: str, gen_code_path: str) -> str:
        """
        Run the stress test using the slow solution and the generator. Return the result of the stress test.
        """
        try:
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
                    print(f"Test case: {gen_output}")
                    print(f"Slow output: {slow_output}")
                    print(f"Fast output: {fast_output}")
                    print("Outputs do not match.")
                    return "Outputs do not match. Stress test failed."
            return "All outputs match. Stress test passed."
        except Exception as e:
            return f"Error: {e}"

    # Tools available to the agent
    tools = [write_to_file, run_stress]
    # Agent to interact with the user
    agent = create_json_chat_agent(llm, tools, chat_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True, handle_validation_errors=True)

    # Run the agent
    #  Try viewing the logs in LangSmith!
    answer = (agent_prompt | {'input': lambda x: x} | agent_executor).invoke(input={'problem_statement': statement})
    print(answer['output'])


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
