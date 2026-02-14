import os
import json # Já está no topo do arquivo.

from google import genai
from fastapi import FastAPI
from duckduckgo_search import DDGS # Esta classe vem do pacote 'ddgs'

# Carrega variáveis de ambiente do arquivo .env
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Configura o modelo Gemini e os parâmetros de geração globalmente
MODEL_NAME = "gemini-1.5-flash-latest"
generation_config = {
    "response_mime_type": "application/json", # Garante que a IA retorne apenas JSON
    "temperature": 0.7, # Controla a criatividade: 0.0 é mais determinístico, 1.0 mais criativo
    "top_p": 0.95,      # Amostragem para diversidade
    "top_k": 60,        # Amostragem para diversidade
    "max_output_tokens": 1024 # Limita o tamanho da resposta da IA para evitar custos excessivos
}
# As configurações de segurança são omitidas para usar as configurações padrão do Gemini,
# que são recomendadas para a maioria da maioria dos casos de uso.

# A inicialização do modelo (gemini_model = client.get_model(MODEL_NAME)) foi removida.
# O modelo será referenciado diretamente na chamada generate_content,
# evitando AttributeError e garantindo que o backend não trave na inicialização.


# --- DEFINIÇÃO DOS PAPÉIS (ROLES) ---

def agente_pesquisador(query: str) -> list[dict]:
    """
    Foca em buscar dados brutos e links oficiais usando a biblioteca duckduckgo-search.
    Retorna uma lista de dicionários, onde cada dicionário contém 'title', 'body' (texto) e 'href' (link).
    """
    search_results = []
    try:
        # Realiza a busca utilizando DDGS().text(). Limitamos a 5 resultados para relevância.
        # O padrão 'with DDGS() as ddgs:' garante que a sessão seja gerenciada corretamente.
        with DDGS() as ddgs:
            # 'wt-br' especifica a região de busca para o Brasil, se desejado.
            for r in ddgs.text(keywords=query, region='wt-br', max_results=5):
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
    Foca em transformar texto em números e código Python usando o Gemini 1.5 Flash.
    Recebe os resultados da busca e gera um insight e dados para visualização em formato JSON.
    """
    # 5. Verifique se o parâmetro `dados` que a função recebe não está vazio.
    if not dados:
        print("DEBUG: O parâmetro 'dados' para agente_analista está vazio. Retornando erro amigável.")
        return {
            "insight": "Nenhum dado bruto recebido para análise. Por favor, forneça dados válidos para que a IA possa analisar.",
            "sugestao_visual": {
                "labels": ["Dados Vazios"],
                "valores": [0],
                "tipo": "bar"
            }
        }
        
    # Não há mais uma verificação global de 'gemini_model'.
    # O cliente 'genai.Client' é inicializado, e quaisquer erros de API Key
    # ou indisponibilidade do modelo serão capturados no bloco try/except abaixo,
    # durante a chamada a 'client.models.generate_content'.
        
    # Combina os corpos e títulos dos resultados da pesquisa em uma única string para o LLM
    context_parts = []
    for i, d in enumerate(dados):
        title = d.get('title', 'Sem Título')
        body = d.get('body', '')
        # Limita o trecho do corpo para evitar prompts excessivamente longos
        snippet = body[:500] + ("..." if len(body) > 500 else "") # Trunca o corpo do texto
        context_parts.append(f"### Documento {i+1}:\nTítulo: {title}\nConteúdo: {snippet}\n")
    
    combined_context = "\n".join(context_parts)
    
    # Retorno de fallback se nenhum conteúdo relevante for encontrado pelo pesquisador
    if not combined_context.strip():
        print("DEBUG: O contexto combinado para o analista está vazio após processamento dos dados. Retornando erro amigável.")
        return {
            "insight": "Nenhum dado relevante ou significativo encontrado para análise quantitativa. A pesquisa não retornou informações úteis.",
            "sugestao_visual": {
                "labels": ["Sem Conteúdo"],
                "valores": [0],
                "tipo": "bar"
            }
        }

    # Cria o prompt detalhado para a IA
    prompt = f"""
    Você é um agente analista de dados altamente qualificado e seu objetivo é extrair inteligência quantitativa e insights acionáveis de um conjunto de dados.
    Recebi os seguintes dados de pesquisa brutos:

    {combined_context}

    Com base nesses dados, por favor, gere uma análise em formato JSON, seguindo estritamente a estrutura abaixo.
    Sua análise deve incluir:

    1.  **insight**: Um resumo executivo conciso e relevante dos dados encontrados.
    2.  **sugestao_visual**: Dados numéricos para um gráfico, com as seguintes propriedades:
        *   **labels**: Uma lista de strings para os eixos do gráfico (mínimo 3 elementos, se possível, representando categorias ou pontos de interesse).
        *   **valores**: Uma lista de números reais (inteiros ou decimais) que correspondam aos `labels`. Tente extrair valores numéricos concretos dos dados fornecidos que possam ser comparados ou que mostrem uma tendência. Se os dados não contiverem números diretamente comparáveis, tente inferir ou agrupar informações quantitativas de forma plausível. É crucial que esta lista contenha **pelo menos 3 pontos de dados numéricos**. Se não for possível extrair ou inferir dados reais, crie 3 valores fictícios plausíveis que correspondam às labels criadas, mas sempre priorize dados reais.
        *   **tipo**: A decisão sobre o tipo de gráfico mais adequado, que pode ser 'bar' para comparações entre categorias ou 'line' para mostrar tendências ao longo do tempo ou sequência. Escolha com base na natureza dos dados extraídos/inferidos.

    Retorne APENAS o objeto JSON. Não inclua texto explicativo adicional antes ou depois do JSON.

    Exemplo do formato de saída esperado:
    ```json
    {{
      "insight": "O mercado de [SETOR] apresentou um crescimento moderado no último trimestre, com investimentos concentrados em [ÁREA_CHAVE] e um desafio persistente em [DESAFIO].",
      "sugestao_visual": {{
        "labels": ["Produto A", "Produto B", "Produto C", "Produto D"],
        "valores": [1200, 850, 1500, 900],
        "tipo": "bar"
      }}
    }}
    ```
    OU
    ```json
    {{
      "insight": "A tendência de adoção da tecnologia X demonstra um crescimento exponencial, com um pico de interesse em [PERÍODO] e uma projeção contínua de alta.",
      "sugestao_visual": {{
        "labels": ["Jan", "Fev", "Mar", "Abr", "Mai"],
        "valores": [10, 25, 60, 150, 300],
        "tipo": "line"
      }}
    }}
    ```
    """
    
    # 1. Adicionar um bloco `try...except` robusto na função `agente_analista`.
    try:
        # Gera o conteúdo usando o modelo Gemini com as configurações definidas.
        # Usando client.models.generate_content diretamente conforme o novo padrão.
        response = client.models.generate_content(
            model=MODEL_NAME, # O modelo é especificado aqui
            contents=prompt,
            generation_config=generation_config,
            # safety_settings são omitidas para usar as configurações padrão do Gemini.
        )
        
        response_text = response.text # Obtém a string JSON diretamente da resposta
        
        # 4. Adicione um `print(f"DEBUG: Resposta da IA: {response.text}")`.
        print(f"DEBUG: Resposta da IA: {response.text}")
        
        # Converte a string JSON para um dicionário Python
        analise_data = json.loads(response_text)
        
        # Validação básica da estrutura da resposta da IA
        if not isinstance(analise_data, dict) or \
           "insight" not in analise_data or \
           not isinstance(analise_data.get('insight'), str) or \
           "sugestao_visual" not in analise_data or \
           not isinstance(analise_data["sugestao_visual"], dict) or \
           "labels" not in analise_data["sugestao_visual"] or \
           not isinstance(analise_data["sugestao_visual"].get('labels'), list) or \
           "valores" not in analise_data["sugestao_visual"] or \
           not isinstance(analise_data["sugestao_visual"].get('valores'), list) or \
           "tipo" not in analise_data["sugestao_visual"] or \
           analise_data["sugestao_visual"].get('tipo') not in ['bar', 'line']:
            raise ValueError("Resposta da IA não está no formato JSON esperado ou contém campos inválidos.")

        # Garante que haja pelo menos 3 pontos para visualização, conforme requisito
        if len(analise_data["sugestao_visual"]["labels"]) < 3 or \
           len(analise_data["sugestao_visual"]["valores"]) < 3 or \
           len(analise_data["sugestao_visual"]["labels"]) != len(analise_data["sugestao_visual"]["valores"]):
            print(f"Aviso: IA retornou {len(analise_data['sugestao_visual']['labels'])} pontos para visualização. Esperado no mínimo 3. Usando fallback padrão para garantir a exibição.")
            # 2. Se o Gemini falhar ou retornar um JSON inválido, a função deve retornar um dicionário padrão com um aviso de erro.
            return {
                "insight": analise_data.get('insight', 'Análise limitada. Dados insuficientes para um gráfico significativo.'),
                "sugestao_visual": {
                    "labels": ["Dado 1", "Dado 2", "Dado 3"],
                    "valores": [10, 20, 15], # Valores de fallback para garantir 3 pontos
                    "tipo": "bar"
                }
            }
            
        return analise_data

    except genai.types.BlockedPromptException as e:
        print(f"Erro no agente analista: Prompt bloqueado pela segurança do Gemini. Detalhes: {e}")
        # 2. Se o Gemini falhar ou retornar um JSON inválido, a função deve retornar um dicionário padrão com um aviso de erro.
        return {
            "insight": "A análise foi bloqueada devido a preocupações de segurança com o prompt ou conteúdo. Por favor, reformule sua pergunta.",
            "sugestao_visual": {
                "labels": ["Segurança Bloqueada"],
                "valores": [0],
                "tipo": "bar"
            }
        }
    except json.JSONDecodeError as e:
        print(f"Erro no agente analista: Falha ao decodificar JSON da resposta da IA. Detalhes: {e}. Resposta bruta da IA: '{response_text[:500]}...'")
        # 2. Se o Gemini falhar ou retornar um JSON inválido, a função deve retornar um dicionário padrão com um aviso de erro.
        return {
            "insight": f"Erro de formato JSON na resposta da IA: A IA não retornou um JSON válido. Por favor, tente novamente.",
            "sugestao_visual": {
                "labels": ["Erro JSON"],
                "valores": [0],
                "tipo": "bar"
            }
        }
    except Exception as e:
        # Este bloco captura erros gerais, incluindo problemas com a API Key do Gemini
        # ou indisponibilidade do serviço.
        print(f"Erro inesperado no agente analista: {e}")
        # 2. Se o Gemini falhar ou retornar um JSON inválido, a função deve retornar um dicionário padrão com um aviso de erro.
        return {
            "insight": f"Erro interno ao gerar análise: {e}. Por favor, verifique sua API Key e conexão e tente novamente.",
            "sugestao_visual": {
                "labels": ["Erro Inesperado"],
                "valores": [0],
                "tipo": "bar"
            }
        }

def agente_auditor(resposta: dict, fontes: list[str]):
    """
    Foca em validar se a resposta não tem mentiras.
    Por enquanto, é um placeholder básico que verifica a presença de insight e fontes.
    """
    # Verifica se a resposta contém um insight e se há fontes para validação.
    # Esta é uma auditoria superficial. Em um cenário real, um LLM faria uma validação cruzada mais profunda.
    if not resposta.get('insight') or "erro" in resposta.get('insight', '').lower() or "bloqueada" in resposta.get('insight', '').lower():
        print("DEBUG: Auditoria falhou: Insight vazio ou contém mensagem de erro.")
        return False
    if not fontes:
        print("DEBUG: Auditoria falhou: Nenhuma fonte encontrada para validação.")
        return False
    
    return True

@app.get("/analise-quantitativa") # Nome do endpoint ajustado para corresponder ao frontend
async def pipeline_analise(pergunta: str):
    """
    Orquestra a chamada dos agentes para realizar a análise da pergunta.
    """
    # 1. Agente Pesquisador: Busca dados brutos e links
    bruto_resultados = agente_pesquisador(pergunta)
    
    if not bruto_resultados:
        print(f"DEBUG: Pipeline de análise: Agente Pesquisador não encontrou resultados para a pergunta: '{pergunta}'")
        return {"error": "Nenhum resultado encontrado pelo agente pesquisador para a sua pergunta. Tente refinar a busca.", "fontes": []}

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
        # Se a auditoria falhar, tenta retornar uma mensagem de erro mais específica
        error_insight = analise.get('insight', '')
        # Ajustado para considerar os novos insights de erro do analista
        if "erro" in error_insight.lower() or "bloqueada" in error_insight.lower() or \
           "indisponível" in error_insight.lower() or "api key" in error_insight.lower() or \
           "dados vazios" in error_insight.lower() or "sem conteúdo" in error_insight.lower() or \
           "formato json" in error_insight.lower():
            # Se o insight já contém uma mensagem de erro do analista
            print(f"DEBUG: Pipeline de análise: Auditoria falhou, insight do analista já indica erro: '{error_insight}'")
            return {"error": error_insight, "fontes": fontes_links}
        else:
            # Erro genérico da auditoria (ex: insight vazio ou sem fontes)
            print(f"DEBUG: Pipeline de análise: Auditoria falhou por motivo genérico. Insight: '{error_insight}', Fontes disponíveis: {bool(fontes_links)}")
            return {"error": "Falha na auditoria da análise. O insight gerado ou as fontes disponíveis não puderam ser validados. Tente novamente ou reformule a pergunta.", "fontes": fontes_links}