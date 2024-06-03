import hashlib

hash = '1a036bd1a439234526bdff1690911814'

def calculate_hash(file_path):
    with open(file_path, 'rb') as file:
        data = file.read()
        return hashlib.md5(data).hexdigest()
hash_n = calculate_hash('ruGPT3/model.safetensors')
print('Модель совпадает с прошлой' if hash_n == hash else f'Модель не совпадает с прошлой\nHash: {hash_n}')