from search import search_prompt
from dotenv import load_dotenv
load_dotenv()

def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Chat pronto! Digite 'sair' para encerrar.\n")

    while True:
        pergunta = input("PERGUNTA: ").strip()

        if pergunta.lower() in ("sair", "exit", "quit"):
            print("Encerrando o chat.")
            break

        if not pergunta:
            continue

        resposta = chain.invoke(pergunta)
        print(f"RESPOSTA: {resposta}\n")

if __name__ == "__main__":
    main()