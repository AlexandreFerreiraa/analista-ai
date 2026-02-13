import os
import requests
import logging
from pathlib import Path

# Configure logging para fornecer feedback detalhado durante a execução
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Funções Auxiliares para Operações de Arquivos ---

def read_file_safely(file_path: Path) -> str | None:
    """
    Lê um arquivo do caminho fornecido, lidando com várias codificações e erros potenciais.
    Retorna o conteúdo do arquivo como uma string, ou None se a leitura falhar
    ou se for detectado um arquivo binário.
    Esta função aprimora o parser de arquivos para ler arquivos com diferentes extensões
    de forma mais segura, conforme solicitado.
    """
    if not file_path.is_file():
        logging.warning(f"Arquivo não encontrado: {file_path}")
        return None

    # Verifica se o arquivo é provavelmente binário para evitar erros de decodificação em arquivos não-texto.
    # Esta é uma heurística; pode não ser perfeita para todos os casos, mas é robusta para a maioria.
    try:
        with open(file_path, 'rb') as f:
            # Lê um pequeno trecho para verificar indicadores comuns de arquivos binários.
            sample = f.read(1024)
            if b'\0' in sample: # Um byte nulo ('\0') frequentemente indica um arquivo binário.
                logging.warning(f"O arquivo {file_path} parece ser binário. Ignorando decodificação de texto.")
                return None
    except Exception as e:
        logging.error(f"Erro ao verificar se {file_path} é binário: {e}")
        return None

    # Tenta decodificar o arquivo como texto usando codificações comuns.
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    for encoding in encodings_to_try:
        try:
            content = file_path.read_text(encoding=encoding)
            logging.info(f"Conteúdo de {file_path} lido com sucesso usando codificação {encoding}.")
            return content
        except UnicodeDecodeError:
            # Se a decodificação falhar, tenta a próxima codificação.
            continue
        except Exception as e:
            # Outros erros durante a leitura são considerados críticos para esta tentativa.
            logging.error(f"Erro ao ler {file_path} com codificação {encoding}: {e}")
            return None

    logging.warning(f"Não foi possível decodificar {file_path} com as codificações de texto comuns. Pode ser um arquivo não-texto ou usar uma codificação de texto não suportada.")
    return None

def write_file_safely(file_path: Path, content: str) -> bool:
    """
    Escreve o conteúdo em um arquivo, criando os diretórios necessários.
    Retorna True em caso de sucesso, False em caso de falha.
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding='utf-8')
        logging.info(f"Escrita em {file_path} realizada com sucesso.")
        return True
    except Exception as e:
        logging.error(f"Falha ao escrever em {file_path}: {e}")
        return False

# --- Interação com a API do GitHub ---

def close_github_issue(issue_number: int, repo_name: str, github_token: str) -> bool:
    """
    Fecha uma issue do GitHub usando a API do GitHub.
    
    Args:
        issue_number: O número da issue a ser fechada.
        repo_name: O nome completo do repositório (ex: "owner/repo").
        github_token: Um token do GitHub com permissão 'issues:write'.
        
    Returns:
        True se a issue foi fechada com sucesso, False caso contrário.
    """
    api_url = f"https://api.github.com/repos/{repo_name}/issues/{issue_number}"
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json"
    }
    payload = {
        "state": "closed"
    }

    logging.info(f"Tentando fechar a issue #{issue_number} no repositório {repo_name}...")
    response = None # Inicializa a resposta para lidar com falhas de requisição
    try:
        response = requests.patch(api_url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()  # Lança uma exceção para erros HTTP (4xx ou 5xx)
        logging.info(f"Issue #{issue_number} fechada com sucesso.")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Falha ao fechar a issue #{issue_number}: {e}")
        if response is not None:
            logging.error(f"Status da resposta: {response.status_code}")
            logging.error(f"Corpo da resposta: {response.text}")
        return False

# --- Lógica Principal do Agente ---
def main():
    logging.info("Agente AI iniciado.")

    # Variáveis de ambiente são cruciais para o contexto do GitHub Actions
    github_token = os.getenv("GITHUB_TOKEN")
    github_repository = os.getenv("GITHUB_REPOSITORY")  # Ex: "octocat/Spoon-Knife"
    # GITHUB_ISSUE_NUMBER não está diretamente disponível em todos os tipos de evento,
    # ele é frequentemente extraído de GITHUB_REF (ex: 'refs/heads/issue-123')
    # ou do payload do evento em si. Para este exercício, assumimos que ele está definido.
    github_issue_number_str = os.getenv("GITHUB_ISSUE_NUMBER") 
    
    if not github_token:
        logging.error("Variável de ambiente GITHUB_TOKEN não definida. Não é possível interagir com a API do GitHub.")
        exit(1)
    if not github_repository:
        logging.error("Variável de ambiente GITHUB_REPOSITORY não definida. Não é possível determinar o repositório.")
        exit(1)
    if not github_issue_number_str:
        logging.error("Variável de ambiente GITHUB_ISSUE_NUMBER não definida. Não é possível fechar uma issue específica.")
        # Em um ambiente real do GitHub Actions, você pode analisar isso a partir de GITHUB_REF ou do payload do evento.
        # Para esta demonstração, devemos sair se estiver faltando, pois fechar uma issue aleatória é perigoso.
        exit(1)
    
    try:
        issue_number = int(github_issue_number_str)
    except ValueError:
        logging.error(f"GITHUB_ISSUE_NUMBER '{github_issue_number_str}' não é um número inteiro válido. Não é possível fechar a issue.")
        exit(1)

    logging.info(f"Agente operando no repositório: {github_repository}, número da issue: {issue_number}")

    # --- SIMULAÇÃO DO PROCESSO DE MODIFICAÇÃO DE ARQUIVOS PELO AGENTE ---
    # Esta seção demonstra como o agente usaria as novas funções `read_file_safely`
    # e `write_file_safely` como parte de seu fluxo de trabalho.
    # Em um cenário real, o agente receberia instruções (ex: de um comentário na issue),
    # as analisaria, decidiria sobre as mudanças, leria os arquivos relevantes,
    # modificaria seu conteúdo e, em seguida, escreveria o conteúdo atualizado de volta para o sistema de arquivos.

    target_file = Path("./backend/main.py") # Exemplo: Agente quer modificar este arquivo

    logging.info(f"Simulando trabalho do agente: tentando ler {target_file}...")
    original_content = read_file_safely(target_file)

    if original_content is None:
        logging.error(f"Não foi possível ler {target_file}. Abortando modificação de arquivo e fechamento da issue.")
        exit(1)

    # Simula a geração de uma mudança: Adiciona um comentário no topo do arquivo.
    # Em um cenário real, o agente geraria um novo conteúdo baseado na análise da requisição.
    new_content = "# Este arquivo foi modificado pelo Agente AI para adicionar uma nova funcionalidade ou correção.\n" \
                  "# O conteúdo original começa abaixo:\n\n" + original_content
    
    logging.info(f"Simulando salvamento das mudanças em {target_file}...")
    files_saved_successfully = write_file_safely(target_file, new_content)

    if files_saved_successfully:
        logging.info("Os arquivos foram simulados como salvos no sistema de arquivos.")
        
        # --- CHAMADA DA FUNÇÃO PARA FECHAR A ISSUE ---
        # Este é o novo requisito: fechar a issue após salvar os arquivos.
        logging.info("Tentando fechar a issue original do GitHub...")
        if close_github_issue(issue_number, github_repository, github_token):
            logging.info("Agente AI concluiu sua tarefa com sucesso e fechou a issue.")
        else:
            logging.error("Agente AI concluiu sua tarefa, mas falhou ao fechar a issue.")
            exit(1) # Considerar sair com erro se o fechamento da issue for crítico
    else:
        logging.error("A simulação de salvamento de arquivos falhou. A issue não será fechada.")
        exit(1) # Sair com erro se o salvamento do arquivo falhar

if __name__ == "__main__":
    main()