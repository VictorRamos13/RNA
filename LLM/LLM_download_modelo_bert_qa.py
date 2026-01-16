from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import os

nome_modelo = 'pierreguillou/bert-base-cased-squad-v1.1-portuguese'
diretorio_local = './modelo_bert_qa_pt'

print('='*60)
print('DOWNLOAD DO MODELO BERT Q&A PORTUGUES')
print('='*60)
print(f'modelo: {nome_modelo}')
print(f'destino: {diretorio_local}')
print('='*60)
print()

#criar diretorio se nao existir
if not os.path.exists(diretorio_local):
   os.makedirs(diretorio_local)

print('baixando tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(nome_modelo)
tokenizer.save_pretrained(diretorio_local)
print('tokenizer salvo')
print()

print('baixando modelo...')
modelo = AutoModelForQuestionAnswering.from_pretrained(nome_modelo)
modelo.save_pretrained(diretorio_local)
print('modelo salvo')
print()

#verificar tamanho
tamanho_total = 0
for raiz, dirs, arquivos in os.walk(diretorio_local):
   for arquivo in arquivos:
      caminho = os.path.join(raiz, arquivo)
      tamanho_total += os.path.getsize(caminho)

print('='*60)
print('DOWNLOAD CONCLUIDO')
print(f'tamanho total: {tamanho_total / 1024 / 1024:.1f} MB')
print(f'local: {os.path.abspath(diretorio_local)}')
print('='*60)
