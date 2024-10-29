import io
import base64
from dataclasses import dataclass, field

from PIL import Image
from openai import OpenAI, ChatCompletion


def encode_base64(image: Image.Image):
    """
    """
    buffer = io.BytesIO()
    image = image.convert('RGB')
    image.save(buffer, format='JPEG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def unpack_content(response: ChatCompletion):
    """
    """
    return response.choices[0].message.content


@dataclass
class ModelGptInput:
    """
    """
    content: list[str] = field(default_factory=list)

    def append(self, input: str | Image.Image):
        """
        """
        def encode_text(text):
            return {'type': 'text', 'text': text}

        def encode_image(image):
            encoded = encode_base64(image)
            return {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{encoded}'}}
    
        if isinstance(input, str):
            self.content.append(encode_text(input))
        elif isinstance(input, Image.Image):
            self.content.append(encode_image(input))
        else:
            raise TypeError(f'ModelGptInput does not support type {type(input)}.')
    
    def extend(self, inputs: list[str | Image.Image]):
        """
        """
        for input in inputs: 
            self.append(input)


class ModelGpt:
    """
    """
    def __init__(self, model='gpt-4o'):
        """
        """
        self.model = model
        self.client = OpenAI()
        self.reset()

    def __call__(self, input: ModelGptInput, max_tokens=512, temperature=0.5) -> dict:
        """
        """        
        self.messages.append({'role': 'user', 'content': input.content})
        response = self.client.chat.completions.create(model=self.model, messages=self.messages, max_tokens=max_tokens, temperature=temperature)
        self.messages.append({'role': 'assistant', 'content': unpack_content(response)})
        return response

    def reset(self):
        """
        """
        self.messages = []


if __name__ == '__main__':
    input = ModelGptInput()
    input.append('Caption this image.')
    input.append(Image.open('tests/test.png'))
    model = ModelGpt(model='gpt-4-turbo')
    response = model(input)
    print(response)
    print()
    print(unpack_content(response))