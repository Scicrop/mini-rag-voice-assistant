import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# Carregar os documentos do diretório 'data'
documents = SimpleDirectoryReader("./docs").load_data()
if not documents:
    raise ValueError("Nenhum documento encontrado no diretório './data'. Certifique-se de que há arquivos Markdown lá.")

# Criar o índice vetorial a partir dos documentos
vector_index = VectorStoreIndex.from_documents(documents)

# Persistir o índice em disco
vector_index.storage_context.persist(persist_dir="index_storage")

print("Índice criado e salvo com sucesso em 'index_storage'.")