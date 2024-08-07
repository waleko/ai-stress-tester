import re
import subprocess
import tempfile
from typing import Tuple, Union, List

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.messages import AIMessage

load_dotenv()


def get_codeforces_statement(codeforces_url: str) -> str:
    """
    This function retrieves the problem statement from the given Codeforces URL.
    """
    response = requests.get(codeforces_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        statement = soup.find('div', class_='problem-statement')
        if statement:
            return statement.get_text(strip=True, separator='\n')
        else:
            raise ValueError('Problem statement not found.')
    else:
        raise ValueError(f'Failed to fetch the URL. Status code: {response.status_code}')


def compile(compile_command: str, source_file: str, output_file: str) -> None:
    """
    This function compiles the source file using the given compile command.
    """
    try:
        subprocess.run(f'{compile_command} {source_file} -o {output_file}', shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f'Failed to compile the source file. Error: {e}')
        exit(1)


def invoke(executable: str, input_text: str) -> Tuple[str, int]:
    with tempfile.TemporaryFile('w+t') as temp_input:
        temp_input.write(input_text)
        temp_input.seek(0)
        try:
            result = subprocess.run(executable, stdin=temp_input, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, check=True)
            return result.stdout, result.returncode
        except subprocess.CalledProcessError as e:
            return e.stderr, e.returncode


def extract_code(obj: Union[AIMessage, str], tag="cpp") -> List[str]:
    if isinstance(obj, AIMessage):
        raw_text = obj.content
    else:
        raw_text = obj
    return re.findall(rf'```{tag}\n(.*?)```', raw_text, re.DOTALL)
