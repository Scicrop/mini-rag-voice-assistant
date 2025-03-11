# Importando as bibliotecas necessárias
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter  # Para personalizar chunks
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # Novo embedding
import chromadb
import argparse
import os

# Função para configurar o RAG
def setup_rag(model_name, embedding_type="ollama"):
    # 1. Configurando o modelo de embedding
    if embedding_type == "ollama":
        embedding_model = OllamaEmbedding(
            model_name=model_name,
            base_url="http://localhost:11434"
        )
    else:  # Usando sentence-transformers como alternativa
        embedding_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
        )

    # 2. Configurando o LLM do Ollama
    llm = Ollama(
        model=model_name,
        base_url="http://localhost:11434",
        temperature=0.7
    )

    # 5. Verificando e lendo os documentos do diretório
    if not os.path.exists("docs") or not os.listdir("docs"):
        raise FileNotFoundError("A pasta 'docs' não existe ou está vazia. Adicione documentos .txt ou .md nela.")
    documents = SimpleDirectoryReader("docs").load_data()

    # 6. Personalizando o tamanho dos chunks
    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)  # Ajuste os valores
    nodes = node_parser.get_nodes_from_documents(documents)

    # 7. Configurando o ChromaDB
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = chroma_client.get_or_create_collection("rag_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 8. Criando o índice com os nós (chunks personalizados)
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embedding_model
    )

    # 9. Criando o query engine
    query_engine = index.as_query_engine(llm=llm)

    return query_engine

# Função para processar a pergunta
def process_query(pergunta, model_name, embedding_type):
    try:
        # Configurando o RAG com o modelo e tipo de embedding especificados
        query_engine = setup_rag(model_name, embedding_type)

        # Fazendo a pergunta ao query engine
        resposta = query_engine.query(pergunta)

        # Imprimindo a pergunta, modelo e resposta
        print(f"Modelo usado: {model_name}")
        print(f"Tipo de embedding: {embedding_type}")
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
    parser.add_argument(
        "--embedding",
        type=str,
        choices=["ollama", "sentence-transformers"],
        default="ollama",
        help="Tipo de embedding a ser usado: 'ollama' ou 'sentence-transformers' (padrão: ollama)"
    )

    # Parseando os argumentos
    args = parser.parse_args()

    # Processando a pergunta com o modelo e embedding fornecidos
    process_query(args.pergunta, args.modelo, args.embedding)

if __name__ == "__main__":
    main()