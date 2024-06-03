from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.cuda import is_available
from view import view
from recognize import recognize

device = 'cuda' if is_available() else 'cpu'

model_path = './ruGPT3'
model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True).to(device)
model.half()

tokenizer = AutoTokenizer.from_pretrained('sberbank-ai/rugpt3medium_based_on_gpt2', low_cpu_mem_usage=True)

def generate(text: str):
    input_ids = tokenizer.encode(f'Формула словами: {text}\nФормула специальными буквами: <s>', return_tensors="pt").to(device)
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=100,     # Максимальное количество новых токенов (слов).
        do_sample=True,         # Используется ли выборка для генерации текста.
        top_k=50,               # Количество наиболее вероятных токенов для рассмотрения.
        top_p=0.98,             # Вероятность рассматриваемых токенов.
        temperature=0.3,        # Разнообразие в генерируемом тексте.
        num_beams=1,            # Количество параллельных путей генерации.
        no_repeat_ngram_size=5  # Предотвращение повторения n-грамм.
    )



    output_text = tokenizer.decode(output[0], skip_special_tokens=True).replace(f'Формула словами: {text}\nФормула специальными буквами: ', '').split('\n\n')[0]
    print(output_text)
    return output_text
try:
    generate('Test')
    print('Модель готова к работе')
except:
    print('Модель недоступна')
    exit()
while True:
    print('Идет распознавание формулы...')
    text = recognize()
    print(text)
    view(generate(text))