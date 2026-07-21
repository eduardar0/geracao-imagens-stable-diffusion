# Geração de Imagens com Stable Diffusion

Este projeto utiliza Inteligência Artificial Generativa para criar imagens a partir de descrições textuais utilizando o modelo Stable Diffusion.

O sistema recebe um prompt descrevendo a imagem desejada, permite definir elementos que devem ser evitados por meio de um negative prompt e gera uma ou mais imagens utilizando a biblioteca Diffusers.

---

## Objetivo do Projeto

O objetivo deste projeto é demonstrar a utilização de um modelo de geração de imagens baseado em texto.

O sistema permite:
.
- Gerar imagens a partir de descrições textuais;
- Definir elementos que não devem aparecer na imagem;
- Gerar múltiplas imagens para um mesmo prompt;
- Controlar a quantidade de etapas de inferência;
- Definir a altura e largura das imagens;
- Utilizar uma seed para reproduzir os resultados;
- Controlar a influência do prompt na geração da imagem;
- Utilizar GPU CUDA quando disponível;
- Utilizar CPU quando uma GPU compatível não estiver disponível;
- Exibir as imagens geradas utilizando Matplotlib.

---

## Tecnologias Utilizadas

- Python 3
- PyTorch
- Hugging Face Diffusers
- Stable Diffusion
- Matplotlib

---

## Bibliotecas Utilizadas

### PyTorch

Utilizada para:

- Verificar a disponibilidade de uma GPU CUDA;
- Definir o dispositivo de execução;
- Criar um gerador aleatório;
- Controlar a seed utilizada na geração das imagens.

---

### Diffusers

A biblioteca `diffusers` fornece ferramentas para trabalhar com modelos de difusão e geração de imagens.

Neste projeto, é utilizado:

```python
StableDiffusionPipeline
```

Essa classe permite carregar um modelo Stable Diffusion e gerar imagens a partir de prompts de texto.

---

### Matplotlib

Utilizada para:

- Criar a estrutura de visualização;
- Exibir as imagens geradas;
- Organizar múltiplas imagens lado a lado.

---

# Modelo Utilizado

O projeto utiliza o modelo:

```text
stabilityai/stable-diffusion-2-1-base
```

O modelo é carregado através da biblioteca Diffusers:

```python
pipeline = StableDiffusionPipeline.from_pretrained(
    pretrained_model_or_path
).to(device)
```

Na primeira execução, o modelo pode ser baixado automaticamente e armazenado localmente para utilização posterior.

---

# Estrutura do Projeto

```text
geracao-imagens-stable-diffusion/
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Funcionamento do Projeto

O projeto possui uma função principal responsável por gerar imagens:

```python
def generate_images(
    prompt,
    negative_prompt,
    num_images_per_prompt,
    num_inference_steps,
    height,
    width,
    seed,
    guidance_scale
):
```

Essa função recebe todos os parâmetros necessários para controlar o processo de geração.

---

# Seleção do Dispositivo

O programa verifica automaticamente se existe uma GPU compatível com CUDA:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Se uma GPU CUDA estiver disponível:

```text
cuda
```

será utilizada.

Caso contrário:

```text
cpu
```

será utilizado.

A utilização de GPU geralmente torna o processo de geração consideravelmente mais rápido.

---

# Prompt

O prompt é a descrição textual da imagem desejada:

```python
prompt = "Generate a high-quality image of a clownfish in an aquarium."
```

Nesse exemplo, o modelo recebe uma instrução para gerar uma imagem de alta qualidade de um peixe-palhaço em um aquário.

O prompt pode ser alterado para descrever diferentes imagens.

Exemplo:

```python
prompt = "A futuristic city at night, highly detailed, cinematic lighting."
```

---

# Negative Prompt

O negative prompt define elementos que devem ser evitados na imagem:

```python
negative_prompt = "It should not contain seahorses"
```

Nesse exemplo, o modelo é instruído a não incluir cavalos-marinhos na imagem gerada.

O negative prompt pode ser utilizado para evitar:

- Objetos;
- Animais;
- Pessoas;
- Características visuais;
- Elementos indesejados;
- Artefatos.

Exemplo:

```python
negative_prompt = "blurry, low quality, distorted, deformed"
```

---

# Quantidade de Imagens

A quantidade de imagens geradas é definida por:

```python
num_images_per_prompt = 2
```

Nesse caso, duas imagens são geradas para o mesmo prompt.

Por exemplo:

```python
num_images_per_prompt = 4
```

irá solicitar quatro imagens.

---

# Número de Etapas de Inferência

O número de etapas utilizadas no processo de geração é definido por:

```python
num_inference_steps = 50
```

Esse valor controla a quantidade de etapas utilizadas pelo modelo para gerar a imagem.

De forma geral:

- Menos etapas podem tornar a geração mais rápida;
- Mais etapas podem aumentar o tempo de processamento;
- Um número maior de etapas pode melhorar o processo de geração até determinado ponto.

---

# Dimensões da Imagem

A altura e a largura da imagem são definidas por:

```python
height = 512
width = 512
```

Nesse caso, as imagens geradas possuem:

```text
512 × 512 pixels
```

Esses valores podem ser alterados conforme a capacidade do hardware e os requisitos do projeto.

---

# Seed

A seed é definida por:

```python
seed = 1234
```

Ela é utilizada para inicializar o gerador aleatório:

```python
generator = torch.Generator(
    device=device
).manual_seed(seed)
```

A utilização da mesma seed, juntamente com os mesmos parâmetros, permite obter resultados reprodutíveis ou semelhantes.

---

# Guidance Scale

O parâmetro:

```python
guidance_scale = 7.5
```

controla a influência do prompt na imagem gerada.

Valores maiores tendem a fazer com que a imagem siga mais rigidamente a descrição fornecida.

Valores menores podem permitir maior liberdade criativa ao modelo.

---

# Geração das Imagens

A geração é realizada através do pipeline:

```python
images = pipeline(
    prompt=prompt,
    num_images_per_prompt=num_images_per_prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=num_inference_steps,
    height=height,
    width=width,
    generator=generator,
    guidance_scale=guidance_scale
)["images"]
```

O resultado é uma lista contendo as imagens geradas:

```python
images
```

---

# Visualização das Imagens

Após a geração, o programa verifica quantas imagens foram produzidas:

```python
total_images = len(images)
```

Em seguida, cria uma estrutura de visualização:

```python
fig, axes = plt.subplots(
    1,
    total_images,
    figsize=(total_images * 10, 10)
)
```

Quando apenas uma imagem é gerada:

```python
if total_images == 1:
    axes.imshow(images[0])
    axes.axis("off")
```

Quando são geradas várias imagens:

```python
else:
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis("off")
```

Por fim, as imagens são exibidas:

```python
plt.show()
```

---

# Fluxo do Projeto

```text
Definição do Prompt
        │
        ▼
Definição do Negative Prompt
        │
        ▼
Configuração dos Parâmetros
        │
        ▼
Verificação da GPU CUDA
        │
        ▼
Carregamento do Stable Diffusion
        │
        ▼
Configuração da Seed
        │
        ▼
Geração das Imagens
        │
        ▼
Organização das Imagens
        │
        ▼
Exibição com Matplotlib
```

---

# Exemplo de Configuração

```python
prompt = "Generate a high-quality image of a clownfish in an aquarium."

negative_prompt = "It should not contain seahorses"

num_images_per_prompt = 2

num_inference_steps = 50

height = 512

width = 512

seed = 1234

guidance_scale = 7.5
```

Essa configuração gera duas imagens de um peixe-palhaço em um aquário, evitando a presença de cavalos-marinhos.

---

# Espaço para Imagem Gerada

Adicione aqui uma imagem gerada pelo projeto:

```markdown
## Resultado da Geração

O modelo foi utilizado para gerar imagens a partir de uma descrição textual e de um negative prompt.

![Imagem gerada pelo Stable Diffusion](images/imagem-gerada.png)
```

---

# Como Configurar o Projeto

## 1. Clonar o Repositório

```bash
git clone https://github.com/SEU_USUARIO/geracao-imagens-stable-diffusion.git

cd geracao-imagens-stable-diffusion
```

---

# 2. Criar um Ambiente Virtual

É recomendado utilizar um ambiente virtual para instalar as dependências do projeto de forma isolada.

## Linux/macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

## Windows

```bash
python -m venv venv

venv\Scripts\activate
```

---

# 3. Instalar as Dependências

Com o ambiente virtual ativado:

```bash
pip install -r requirements.txt
```

O arquivo `requirements.txt` deve conter as dependências necessárias:

```text
torch
diffusers
transformers
accelerate
matplotlib
```

Dependendo do ambiente e do suporte à GPU, a instalação do PyTorch pode exigir uma configuração específica para a versão do CUDA instalada.

---

# 4. Executar o Projeto

No Linux/macOS:

```bash
python3 main.py
```

No Windows:

```bash
python main.py
```

---

# Funcionamento da Execução

Ao executar o programa, ele irá:

1. Verificar se uma GPU CUDA está disponível;
2. Definir o dispositivo de execução;
3. Carregar o modelo Stable Diffusion;
4. Criar um gerador aleatório com a seed definida;
5. Processar o prompt;
6. Processar o negative prompt;
7. Gerar a quantidade de imagens solicitada;
8. Organizar as imagens para visualização;
9. Exibir as imagens geradas na tela.

---

# Exemplo de Resultado

O programa gera imagens a partir da descrição fornecida no prompt.

Por exemplo:

```text
Prompt:
Generate a high-quality image of a clownfish in an aquarium.

Negative Prompt:
It should not contain seahorses
```

O modelo então gera duas imagens baseadas na descrição fornecida.

---

# Observações

- Na primeira execução, o modelo pode ser baixado automaticamente.
- O download do modelo pode exigir uma quantidade significativa de espaço em disco.
- A geração de imagens pode consumir bastante memória RAM e memória de GPU.
- A execução em uma GPU CUDA geralmente é mais rápida do que a execução em CPU.
- O tempo de geração depende do hardware disponível e do número de etapas de inferência.
- Imagens maiores podem exigir mais memória.
- A seed pode ser alterada para gerar diferentes resultados.
- O prompt influencia diretamente o conteúdo da imagem gerada.
- O negative prompt pode ser utilizado para tentar evitar determinados elementos.
- Os resultados de modelos generativos podem variar mesmo quando parâmetros semelhantes são utilizados.

---

# Autor

Projeto desenvolvido para fins de estudo e prática de Inteligência Artificial Generativa, modelos de difusão e geração de imagens utilizando Python, PyTorch, Hugging Face Diffusers e Stable Diffusion.# geracao-imagens-stable-diffusion
