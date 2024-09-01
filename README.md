# Algoritmos Evolutivos: Comparação entre GA e PSO

Este repositório contém o código-fonte dos experimentos desenvolvidos para o artigo que compara o Algoritmo Genético (GA) e a Otimização por Enxame de Partículas (PSO) em diferentes cenários de otimização. Os experimentos foram realizados utilizando a linguagem Python, com a biblioteca Streamlit para a interface interativa, além das bibliotecas Mealpy e Opfunu para a implementação dos algoritmos evolutivos.

## Estrutura do Repositório

- `src/exp_1.py`: Script para reproduzir o Experimento #1, que foca na otimização com viés central e viés zero utilizando GA e PSO.
- `src/exp_2.py`: Script para reproduzir o Experimento #2, que aplica estratégias de penalização adaptativa no problema de mola sob tração/compressão utilizando GA e PSO.
- `src/requirements.txt`: Arquivo listando todas as dependências necessárias para executar os scripts.
- `results/`: Pasta onde estão os resultados obtidos após a execução dos experimentos do artigo desenvolvido.


## Pré-requisitos

Para reproduzir os experimentos, você precisará ter o Python 3.8+ instalado em seu ambiente. Recomenda-se utilizar um ambiente virtual para gerenciar as dependências.

## Instalação

1. Clone este repositório em sua máquina local:

    ```bash
    git clone https://github.com/marcelocorni/evolo_cec_apm.git
    cd evolo_cec_apm
    ```

2. Crie um ambiente virtual e ative-o:

    ```bash
    python -m venv venv
    source venv/bin/activate  # Para Linux/MacOS
    venv\Scripts\activate  # Para Windows
    ```

3. Instale as dependências listadas no `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

## Executando os Experimentos

### Experimento #1

O script `src/exp_1.py` permite reproduzir o primeiro experimento, que analisa a eficácia dos algoritmos GA e PSO em problemas de otimização com viés central e viés zero.

Para executar o experimento:

```bash
cd src
streamlit run exp_1.py
```

### Experimento #2

O script `src/exp_2.py` permite reproduzir o segundo experimento, que aplica estratégias de penalização adaptativa ao problema de mola sob tração/compressão.

Para executar o experimento:

```bash
cd src
streamlit run exp_2.py
```

## Contribuições

Contribuições são bem-vindas! Se você tiver sugestões de melhorias ou encontrar problemas, fique à vontade para abrir uma issue ou enviar um pull request.

## Contato

Para mais informações ou para acesso ao artigo, entre em contato através de [marcelo.corni@gmail.com](mailto:marcelo.corni@gmail.com).