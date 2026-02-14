import os
import sys
import urllib.request
import json
from google import genai

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def close_issue():
    issue_number = os.getenv("GITHUB_ISSUE_NUMBER")
    repo = os.getenv("GITHUB_REPOSITORY")
    token = os.getenv("GITHUB_TOKEN")

    if not issue_number or not repo or not token:
        print("⚠️ Info: Dados insuficientes para fechar a issue automaticamente.")
        return

    url = f"https://api.github.com/repos/{repo}/issues/{issue_number}"
    data = json.dumps({"state": "closed", "state_reason": "completed"}).encode("utf-8")
    
    req = urllib.request.Request(url, data=data, method="PATCH")
    req.add_header("Authorization", f"token {token}")
    req.add_header("Content-Type", "application/json")
    
    try:
        with urllib.request.urlopen(req) as resp:
            print(f"✅ Issue #{issue_number} fechada com sucesso.")
    except Exception as e:
        print(f"❌ Erro ao fechar issue: {e}")

def get_project_files():
    files_context = ""
    for root, dirs, files in os.walk("."):
        if any(x in root for x in ["venv", ".git", "__pycache__", "node_modules"]): continue
        for file in files:
            if file.endswith((".py", ".txt", ".yml", ".md")):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        files_context += f"\n\n--- FILE: {path} ---\n{f.read()}"
                except: continue
    return files_context

def main():
    issue_title = os.getenv("ISSUE_TITLE", "Update")
    issue_body = os.getenv("ISSUE_BODY", "")
    projeto = get_project_files()

    prompt = f"VOCÊ É UM ENGENHEIRO SENIOR. Objetivo: {issue_title}\nDetalhes: {issue_body}\n\nPROJETO ATUAL:\n{projeto}\n\nRetorne APENAS os arquivos alterados no formato:\n--- START_FILE: caminho ---\ncodigo\n--- END_FILE ---"

    print("🤖 IA processando...")
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)

    content = response.text
    if "--- START_FILE:" in content:
        for part in content.split("--- START_FILE: ")[1:]:
            header, rest = part.split(" ---",1)
            path = header.strip()
            code = rest.split("--- END_FILE ---")[0].strip()
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"📝 {path} atualizado.")
    
    close_issue()

if __name__ == "__main__":
    main()
