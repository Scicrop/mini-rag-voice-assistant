import os
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI

# Configurar a chave da API do OpenAI
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("A variável de ambiente OPENAI_API_KEY não está definida.")

# Carregar o índice salvo
persist_dir = "index_storage"

storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
vector_index = load_index_from_storage(storage_context)

# Configurar o modelo da OpenAI
llm = OpenAI(model="gpt-4o", api_key=openai_api_key)

# Criar o mecanismo de consulta
query_engine = vector_index.as_query_engine(llm=llm)

# Fazer a consulta
response = query_engine.query("Meu trator não liga, o que devo fazer?")
print(response)