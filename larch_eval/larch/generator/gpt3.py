from typing import Tuple

import requests
from requests.adapters import HTTPAdapter, Retry
from transformers import GPT2Tokenizer

from .base import BaseGenerator


class ChatModelGenerator(BaseGenerator):
    ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions'

    def __init__(self, model: str, key: str):
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        super().__init__(tokenizer, 4097, False)
        self.model = model
        self.api_key = key

    def _create_header(self) -> dict:
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        return headers

    @staticmethod
    def _create_session():
        session = requests.Session()
        retries = Retry(total=5,
                        backoff_factor=0.1,
                        status_forcelist=[500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session

    def generate(self, context: str, prompt: str, max_length: int) -> Tuple[str, str]:
        headers = self._create_header()
        messages = [
            # {"role": "system", "content": context},
            {"role": "user", "content": context + prompt}
        ]
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_length,
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "stream": False,
        }
        session = self._create_session()
        r = session.post(self.ENDPOINT, json=data, headers=headers)
        r.raise_for_status()

        text = r.json()['choices'][0]['message']['content']
        return prompt + text, text

    def calculate_perplexity(self, *args, **kwargs):
        raise NotImplementedError()


class GPT4Generator(ChatModelGenerator):
    def __init__(self, key: str):
        super().__init__(model='gpt-4.1', key=key)


class ClaudeGenerator(ChatModelGenerator):
    def __init__(self, key: str):
        super().__init__(model='anthropic/claude-sonnet-4', key=key)


class GemmaGenerator(ChatModelGenerator):
    def __init__(self, key: str):
        super().__init__(model='google/gemma-3-27b-it', key=key)
