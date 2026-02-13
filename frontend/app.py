import streamlit as st
import requests
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Portal de Transparência AI", layout="wide")

st.title("🔍 Analista de Dados de Utilidade Pública")
st.markdown("---")

with st.sidebar:
    st.header("Consulta")
    pergunta = st.text_area("O que você deseja investigar?", height=150)
    botao = st.button("Analisar")

if botao and pergunta:
    with st.spinner("Consultando bases de dados oficiais..."):
        try:
            # Chama o backend
            response = requests.get(f"http://api:8000/analise-quantitativa?pergunta={pergunta}")
            dados = response.json()
            
            # --- SEÇÃO 1: INSIGHT ---
            st.subheader("💡 Conclusão do Analista")
            st.write(dados.get('insight', 'Sem resumo disponível.'))

            # --- SEÇÃO 2: GRÁFICO REAL ---
            st.subheader("📊 Representação dos Dados")
            visual = dados.get('sugestao_visual', {})
            
            if visual.get('labels') and visual.get('valores'):
                df_grafico = pd.DataFrame({
                    "Categoria": visual['labels'],
                    "Valor": visual['valores']
                })
                
                if visual['tipo'] == 'line':
                    fig = px.line(df_grafico, x="Categoria", y="Valor", markers=True, template="plotly_white")
                else:
                    fig = px.bar(df_grafico, x="Categoria", y="Valor", color="Valor", template="plotly_white")
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Dados insuficientes para gerar o gráfico automático.")

            # --- SEÇÃO 3: FONTES E AUDITORIA (O MAIS IMPORTANTE) ---
            st.markdown("---")
            st.subheader("🔗 Fontes e Bases de Dados (Máx. 5)")
            
            fontes = dados.get('fontes', [])
            if fontes:
                for link in fontes[:5]: # Garante no máximo 5
                    st.markdown(f"- [Acessar Base de Dados/Fonte]({link})")
            else:
                st.write("Nenhuma fonte direta encontrada no contexto.")

        except Exception as e:
            st.error(f"Erro na comunicação: {e}")
            st.info("Dica: Verifique se o backend está rodando e se a API Key do Gemini está correta.")