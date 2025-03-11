# Importando as bibliotecas necessárias
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
import chromadb
import argparse

# Função para configurar o RAG
def setup_rag(model_name):
    # 1. Configurando o modelo de embedding do Ollama
    embedding_model = OllamaEmbedding(
        model_name=model_name,
        base_url="http://localhost:11434"  # URL padrão do Ollama local
    )

    # 2. Configurando o LLM do Ollama
    llm = Ollama(
        model=model_name,
        base_url="http://localhost:11434",
        temperature=0.7
    )

    # 5. Lendo os documentos do diretório
    documents = SimpleDirectoryReader("docs").load_data()

    # 6. Configurando o ChromaDB
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("rag_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 7. Criando o índice com os documentos
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embedding_model
    )

    # 8. Criando o query engine
    query_engine = index.as_query_engine(llm=llm)

    return query_engine

# Função para processar a pergunta
def process_query(pergunta, model_name):
    try:
        # Configurando o RAG com o modelo especificado
        query_engine = setup_rag(model_name)

        # Fazendo a pergunta ao query engine
        resposta = query_engine.query(pergunta)

        # Imprimindo a pergunta, modelo e resposta
        print(f"Modelo usado: {model_name}")
        print(f"Pergunta: {pergunta}")
        print(f"Resposta: {resposta.response}")

    except Exception as e:
        print(f"Erro: {str(e)}")

# Função principal para lidar com argumentos da linha de comando
def main():
    # Configurando o parser de argumentos
    parser = argparse.ArgumentParser(description="Faça uma pergunta ao RAG via linha de comando.")
    parser.add_argument("pergunta", type=str, help="A pergunta que você deseja fazer ao RAG")
    parser.add_argument(
        "--modelo",
        type=str,
        default="cnmoro/Qwen2.5-0.5B-Portuguese-v2:fp16",
        help="Nome do modelo Ollama a ser usado (padrão: cnmoro/Qwen2.5-0.5B-Portuguese-v2:fp16)"
    )

    # Parseando os argumentos
    args = parser.parse_args()

    # Processando a pergunta com o modelo fornecido
    process_query(args.pergunta, args.modelo)

if __name__ == "__main__":
    main()