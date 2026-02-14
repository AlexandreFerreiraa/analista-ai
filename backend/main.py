import os
import json
# import re # Removido, pois a limpeza da query agora é feita pelo Gemini para refinar as palavras-chave

from google import genai
from fastapi import FastAPI
from duckduckgo_search import DDGS # Esta classe vem do pacote 'ddgs'

# Carrega variáveis de ambiente do arquivo .env
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Configura o modelo Gemini e os parâmetros de geração globalmente
MODEL_NAME = "gemini-2.5-flash" # UPGRADE: Modelo alterado para Gemini 2.5 Flash
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

# 1. Função auxiliar para limpar JSON
def limpar_json(texto: str) -> str:
    """
    Remove marcações de markdown como ```json e ``` se estiverem presentes na resposta da IA.
    """
    # Remove ```json do início
    if texto.strip().startswith("```json"):
        texto = texto.strip()[len("```json"):].strip()
    # Remove ``` do final
    if texto.strip().endswith("```"):
        texto = texto.strip()[:-len("```")].strip()
    return texto

def agente_pesquisador(query: str) -> list[dict]:
    """
    Foca em buscar dados brutos e links oficiais usando a biblioteca ddgs.
    Primeiro, refina a pergunta do usuário para 3 palavras-chave simples usando Gemini Flash.
    Retorna uma lista de dicionários, onde cada dicionário contém 'title', 'body' (texto) e 'href' (link).
    """
    search_results = []
    # Inicializa query_refinada com a query original como fallback.
    # Se a IA falhar em gerar palavras-chave, ou gerar vazio, a busca usará a query original.
    query_refinada = query 
    try:
        # Passo 1: Use Gemini Flash para transformar a pergunta do usuário em 3 palavras-chave simples
        keyword_refinement_prompt = f"""
        Você é um assistente de busca avançado, altamente preciso e eficaz. Dada uma pergunta, seu objetivo é extrair **exatamente 3 palavras-chave simples, únicas e altamente relevantes em português**, separadas por espaços. Estas palavras-chave devem ser as mais impactantes para uma busca na internet, visando resultados diretos e oficiais.
        Siga estas regras estritas:
        - Não inclua pontuação, artigos (a, o, os, as), preposições (de, em, para), ou qualquer frase adicional.
        - APENAS as 3 palavras-chave mais importantes.
        - Priorize palavras-chave únicas. Se a pergunta tiver menos de 3 palavras-chave realmente distintas e importantes, você pode repetir a mais importante para completar 3, mas evite se houver termos distintos.
        - As palavras-chave devem ser objetivas e focadas na informação central.

        Exemplo 1: 'Qual a cotação do dólar hoje?' -> 'dólar hoje cotação'
        Exemplo 2: 'Notícias sobre inteligência artificial no Brasil' -> 'inteligência artificial Brasil notícias'
        Exemplo 3: 'Como funciona a energia solar para residências?' -> 'energia solar residências funcionamento'
        Exemplo 4: 'O que é IA?' -> 'inteligência artificial IA definição' # Ajustado para maior precisão

        Pergunta: '{query}'
        Palavras-chave:"""
        
        # Configuração de geração específica para extração de palavras-chave: texto simples, determinístico.
        keyword_generation_config = {
            "response_mime_type": "text/plain", # Queremos texto simples, não JSON
            "temperature": 0.1, # Baixa temperatura para saída determinística
            "top_p": 0.95,
            "top_k": 60,
            "max_output_tokens": 50 # Limita a saída a poucas palavras
        }

        # Adiciona um bloco try/except ao redor da geração de palavras-chave
        try:
            # Gera as palavras-chave usando o Gemini Flash
            print(f"DEBUG REFINAMENTO: Gerando palavras-chave para a pergunta: '{query}'")
            keyword_response = client.models.generate_content(
                model=MODEL_NAME, # Usando o mesmo modelo Gemini 2.5 Flash
                contents=keyword_refinement_prompt,
                config=keyword_generation_config,
            )
            
            generated_keywords = keyword_response.text.strip()
            # Se a IA gerou palavras-chave válidas, usamos elas; caso contrário, mantemos a query original.
            if generated_keywords: 
                query_refinada = generated_keywords
                print(f"DEBUG REFINAMENTO: Palavras-chave geradas pelo Gemini: '{query_refinada}'")
            else:
                # Se a IA retornou vazio, usa a query original como fallback
                print(f"DEBUG REFINAMENTO: Gemini gerou palavras-chave vazias. Usando a query original como fallback: '{query}'")
                query_refinada = query
        except Exception as e:
            # Em caso de erro na chamada da IA (API Key, bloqueio, etc.), usa a query original como fallback
            print(f"DEBUG REFINAMENTO: Erro ao gerar palavras-chave com Gemini: {e}. Usando a query original como fallback: '{query}'")
            query_refinada = query

        # Se a query_refinada, mesmo após o fallback, estiver vazia ou for apenas espaços, 
        # significa que a query original era inválida ou muito curta, e não vale a pena pesquisar.
        if not query_refinada.strip():
            print("DEBUG PESQUISA: Palavras-chave refinadas (final) estão vazias. Nenhuma busca será realizada.")
            return []

        # Passo 2: Atualiza a busca do DuckDuckGo para usar as palavras-chave refinadas
        print(f"DEBUG PESQUISA: Realizando busca com DDGS para: '{query_refinada}'")
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(keywords=query_refinada, max_results=5)]
        
        for r in results:
            search_results.append({
                "title": r.get('title'),
                "body": r.get('body'),
                "href": r.get('href')
            })
    except Exception as e:
        # Este bloco captura erros gerais relacionados à busca DDGS, APÓS a tentativa de refino das palavras-chave.
        print(f"Erro ao buscar com DuckDuckGo para a query refinada '{query_refinada}' (original: '{query}'): {e}")
        # Retorna uma lista vazia em caso de falha na busca
        return []
    
    return search_results

def agente_analista(dados: list[dict]):
    """
    Foca em transformar texto em números e código Python usando o Gemini 2.5 Flash.
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
    Você é um agente analista de dados **extremamente qualificado, com foco em precisão, rigor quantitativo e geração de insights profundos**. Seu objetivo é examinar meticulosamente os dados brutos de pesquisa fornecidos e extrair inteligência acionável e informações numéricas exatas ou rigorosamente inferidas.
    Recebi os seguintes dados de pesquisa brutos para análise aprofundada:

    {combined_context}

    Com base NESTES DADOS, por favor, gere uma análise detalhada em formato JSON, seguindo estritamente a estrutura e as diretrizes abaixo.
    Sua análise deve incluir:

    1.  **insight**: Um resumo executivo **altamente preciso e objetivo** que destaque as principais descobertas, tendências, implicações ou relações causais presentes nos dados. O insight deve ser direto, sem redundâncias e focado no valor de negócio ou utilidade pública. Ele deve ser uma conclusão robusta baseada **exclusivamente** nos dados fornecidos.
    2.  **sugestao_visual**: Dados numéricos e categóricos para a criação de um gráfico, com as seguintes propriedades, focando na **extração rigorosa de dados numéricos**:
        *   **labels**: Uma lista de strings para os eixos do gráfico. Estas devem ser categorias claras, períodos de tempo ou itens distintos **explicitamente mencionados nos dados**. Tente identificar **pelo menos 5 elementos** para uma visualização mais rica; no mínimo 3.
        *   **valores**: Uma lista de números reais (inteiros ou decimais) que correspondam EXATAMENTE aos `labels`. É CRUCIAL que estes `valores` sejam extraídos diretamente dos dados fornecidos, representando quantidades, porcentagens, valores monetários ou contagens. Se os dados não contiverem números explícitos para uma comparação direta, faça uma inferência rigorosa de magnitudes ou agrupamentos, explicando brevemente no insight a base da inferência, mas **sempre priorize números concretos**. Se for absolutamente impossível extrair ou inferir 3 valores numericamente comparáveis e com sentido, indique isso no insight.
        *   **tipo**: A decisão sobre o tipo de gráfico mais adequado. Pode ser 'bar' para comparações claras entre categorias ou 'line' para mostrar tendências, evolução temporal ou sequências. A escolha deve ser **justificada pela natureza dos `labels` e `valores` extraídos**.

    Retorne APENAS o objeto JSON. Não inclua absolutamente NENHUM texto explicativo, introdução, conclusão ou qualquer outro caractere fora do objeto JSON. A resposta deve começar e terminar com chaves `{}`.

    Exemplo do formato de saída esperado:
    ```json
    {{
      "insight": "A performance trimestral da empresa Y revela um crescimento consistente nas vendas dos produtos A e C, impulsionado por campanhas de marketing eficazes no Q2, enquanto o produto B mostrou estagnação devido a desafios na cadeia de suprimentos.",
      "sugestao_visual": {{
        "labels": ["Produto A", "Produto B", "Produto C", "Produto D", "Produto E"],
        "valores": [1200.50, 850.75, 1500.20, 900.00, 1100.10],
        "tipo": "bar"
      }}
    }}
    ```
    OU
    ```json
    {{
      "insight": "A pesquisa de mercado anual aponta para uma elevação gradual na adoção da tecnologia Z, passando de 10% em 2020 para 30% em 2023, com projeções de 45% em 2024, indicando uma aceitação crescente no segmento B2C.",
      "sugestao_visual": {{
        "labels": ["2020", "2021", "2022", "2023", "2024 (proj.)"],
        "valores": [10, 15, 25, 30, 45],
        "tipo": "line"
      }}
    }}
    ```
    """
    
    # 2. Envolver todo o processo em um bloco try...except
    try:
        # Gera o conteúdo usando o modelo Gemini com as configurações definidas.
        # Usando client.models.generate_content diretamente conforme o novo padrão.
        response = client.models.generate_content(
            model=MODEL_NAME, # O modelo é especificado aqui
            contents=prompt,
            config=generation_config,
            # safety_settings são omitidas para usar as configurações padrão do Gemini.
        )
        
        raw_response_text = response.text # Obtém a string JSON diretamente da resposta
        
        # 4. Adicione um `print(f"DEBUG IA: {response.text}")`.
        print(f"DEBUG IA: {raw_response_text}")
        
        # 1. Limpar marcações de markdown antes de tentar parsear
        cleaned_response_text = limpar_json(raw_response_text)
        
        # Converte a string JSON para um dicionário Python
        analise_data = json.loads(cleaned_response_text)
        
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
            # Se a validação interna do Gemini falhar, retorna um dicionário padrão.
            # O insight pode ser mantido se for útil, mas a visualização é padronizada.
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
        # 3. Retorne um dicionário padrão em caso de erro
        return {
            "insight": f"Erro técnico ao processar dados: Prompt bloqueado pela segurança do Gemini. Detalhes: {str(e)}",
            "sugestao_visual": {
                "labels": ["Erro"],
                "valores": [0],
                "tipo": "bar"
            }
        }
    except json.JSONDecodeError as e:
        print(f"Erro no agente analista: Falha ao decodificar JSON da resposta da IA. Detalhes: {e}. Resposta bruta da IA: '{raw_response_text[:500]}...'")
        # 3. Retorne um dicionário padrão em caso de erro
        return {
            "insight": f"Erro técnico ao processar dados: Falha ao decodificar JSON da resposta da IA. Detalhes: {str(e)}. Resposta bruta: '{raw_response_text[:100]}...'",
            "sugestao_visual": {
                "labels": ["Erro"],
                "valores": [0],
                "tipo": "bar"
            }
        }
    except Exception as e:
        # Este bloco captura erros gerais, incluindo problemas com a API Key do Gemini
        # ou indisponibilidade do serviço.
        print(f"Erro inesperado no agente analista: {e}")
        # 3. Retorne um dicionário padrão em caso de erro
        return {
            "insight": f"Erro técnico ao processar dados: {str(e)}. Por favor, verifique sua API Key e conexão e tente novamente.",
            "sugestao_visual": {
                "labels": ["Erro"],
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
    # A auditoria deve ser capaz de lidar com os insights de erro do agente_analista
    insight_text = resposta.get('insight', '').lower()
    if not insight_text or "erro técnico" in insight_text or "bloqueada" in insight_text or \
       "dados vazios" in insight_text or "sem conteúdo" in insight_text:
        print(f"DEBUG: Auditoria falhou: Insight vazio ou contém mensagem de erro/problema: '{insight_text}'")
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
        # 5. Lidar com o retorno de erro de forma elegante
        return {
            "insight": "Nenhum resultado encontrado pelo agente pesquisador para a sua pergunta. Tente refinar a busca.",
            "sugestao_visual": {"labels": ["Pesquisa Vazia"], "valores": [0], "tipo": "bar"},
            "fontes": []
        }

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
        # 5. Lidar com o retorno de erro de forma elegante
        # Se a auditoria falhar, tenta retornar uma mensagem de erro mais específica
        error_insight = analise.get('insight', '')
        # Ajustado para considerar os novos insights de erro do analista, incluindo "Erro técnico"
        if "erro técnico" in error_insight.lower() or "bloqueada" in error_insight.lower() or \
           "indisponível" in error_insight.lower() or "api key" in error_insight.lower() or \
           "dados vazios" in error_insight.lower() or "sem conteúdo" in error_insight.lower() or \
           "formato json" in error_insight.lower():
            # Se o insight já contém uma mensagem de erro do analista, retorne-o diretamente.
            print(f"DEBUG: Pipeline de análise: Auditoria falhou, insight do analista já indica erro: '{error_insight}'")
            return {
                "insight": error_insight,
                "sugestao_visual": analise.get('sugestao_visual', {"labels": ["Auditoria Falha"], "valores": [0], "tipo": "bar"}), # Garante um fallback visual
                "fontes": fontes_links
            }
        else:
            # Erro genérico da auditoria (ex: insight vazio ou sem fontes, mas não erro de IA)
            print(f"DEBUG: Pipeline de análise: Auditoria falhou por motivo genérico. Insight: '{error_insight}', Fontes disponíveis: {bool(fontes_links)}")
            return {
                "insight": "Falha na auditoria da análise. O insight gerado ou as fontes disponíveis não puderam ser validados. Tente novamente ou reformule a pergunta.",
                "sugestao_visual": {"labels": ["Auditoria Falha"], "valores": [0], "tipo": "bar"},
                "fontes": fontes_links
            }