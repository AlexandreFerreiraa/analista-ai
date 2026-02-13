import os
from google import genai
from fastapi import FastAPI
from duckduckgo_search import DDGS

app = FastAPI()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# --- DEFINIÇÃO DOS PAPÉIS (ROLES) ---

def agente_pesquisador(query: str) -> list[dict]:
    """
    Foca em buscar dados brutos e links oficiais usando a biblioteca duckduckgo-search.
    Retorna uma lista de dicionários, onde cada dicionário contém 'title', 'body' (texto) e 'href' (link).
    """
    search_results = []
    try:
        # Realiza a busca utilizando DDGS().text(). Limitamos a 5 resultados para relevância.
        # 'wt-br' especifica a região de busca para o Brasil, se desejado.
        for r in DDGS().text(keywords=query, region='wt-br', max_results=5):
            search_results.append({
                "title": r.get('title'),
                "body": r.get('body'),
                "href": r.get('href')
            })
    except Exception as e:
        print(f"Erro ao buscar com DuckDuckGo para a query '{query}': {e}")
        # Retorna uma lista vazia em caso de falha na busca
        return []
    
    return search_results

def agente_analista(dados: list[dict]):
    """
    Foca em transformar texto em números e código Python.
    Por enquanto, é um placeholder que simula a saída esperada pelo frontend.
    """
    # Concatena os corpos dos resultados da pesquisa para simular dados brutos a serem analisados
    combined_text_sample = " ".join([d.get('body', '') for d in dados if d.get('body')])[:500] + "..."
    
    # Este é um placeholder básico. Em um cenário real, este agente usaria um LLM
    # para analisar 'combined_text_sample' e gerar um insight e dados para visualização.
    return {
        "insight": f"Análise inicial dos dados sobre a pergunta. Conteúdo bruto encontrado: {combined_text_sample}",
        "sugestao_visual": {
            "labels": ["Exemplo A", "Exemplo B", "Exemplo C"],
            "valores": [15, 25, 10],
            "tipo": "bar" # Pode ser 'bar' ou 'line'
        }
    }

def agente_auditor(resposta: dict, fontes: list[str]):
    """
    Foca em validar se a resposta não tem mentiras.
    Por enquanto, é um placeholder básico que verifica a presença de insight e fontes.
    """
    # Verifica se a resposta contém um insight e se há fontes para validação.
    if not resposta.get('insight'):
        return False
    if not fontes:
        return False
    
    # Uma implementação real usaria um LLM para cruzar informações e validar a veracidade.
    # Ex: client.generate_content(f"A conclusão '{resposta['insight']}' é suportada pelas fontes: {', '.join(fontes)}? Responda apenas com 'sim' ou 'não'.")
    return True

@app.get("/analise-quantitativa") # Nome do endpoint ajustado para corresponder ao frontend
async def pipeline_analise(pergunta: str):
    """
    Orquestra a chamada dos agentes para realizar a análise da pergunta.
    """
    # 1. Agente Pesquisador: Busca dados brutos e links
    bruto_resultados = agente_pesquisador(pergunta)
    
    if not bruto_resultados:
        return {"error": "Nenhum resultado encontrado pelo agente pesquisador.", "fontes": []}

    # Extrai apenas os links para o auditor e para a resposta final
    fontes_links = [res.get('href') for res in bruto_resultados if res.get('href')]
    
    # 2. Agente Analista: Transforma os dados brutos em insight e sugestão de visualização
    analise = agente_analista(bruto_resultados)
    
    # 3. Agente Auditor: Valida a análise com base nas fontes
    auditoria_ok = agente_auditor(analise, fontes_links)
    
    if auditoria_ok:
        # Prepara a resposta final no formato esperado pelo frontend
        return {
            "insight": analise.get('insight'),
            "sugestao_visual": analise.get('sugestao_visual'),
            "fontes": fontes_links # Retorna os links extraídos
        }
    else:
        return {"error": "Falha na auditoria da análise.", "fontes": fontes_links}