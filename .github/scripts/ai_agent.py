import os
import sys
from google import genai

# Configuração do Cliente Gemini
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_project_files():
    """Lê os arquivos atuais do projeto para dar contexto à IA."""
    files_context = ""
    for root, dirs, files in os.walk("."):
        if any(x in root for x in ["venv", ".git", "__pycache__"]): continue
        for file in files:
            if file.endswith((".py", ".txt", ".yml", ".md")):
                path = os.path.join(root, file)
                with open(path, "r") as f:
                    files_context += f"\n\n--- FILE: {path} ---\n{f.read()}"
    return files_context

def main():
    issue_title = os.getenv("ISSUE_TITLE", "Melhoria geral")
    issue_body = os.getenv("ISSUE_BODY", "Sem descrição")
    projeto = get_project_files()

    prompt = f"""
    VOCÊ É UM ENGENHEIRO DE SOFTWARE SENIOR.
    OBJETIVO: Resolver a seguinte solicitação: "{issue_title}"
    DETALHES: {issue_body}

    CÓDIGO ATUAL:
    {projeto}

    REGRAS:
    1. Responda APENAS com o código completo dos arquivos que precisam de alteração.
    2. Use o formato:
    --- START_FILE: caminho/do/arquivo ---
    (código)
    --- END_FILE ---
    3. Seja preciso e não quebre a estrutura do Docker.
    """

    print("🤖 IA analisando o projeto...")
    response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=prompt
    )

    # Lógica simples para extrair e sobrescrever os arquivos
    content = response.text
    for part in content.split("--- START_FILE: ")[1:]:
        header, rest = part.split(" ---",1)
        path = header.strip()
        code = rest.split("--- END_FILE ---")[0].strip()
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(code)
        print(f"✅ Arquivo {path} atualizado!")

if __name__ == "__main__":
    main()

