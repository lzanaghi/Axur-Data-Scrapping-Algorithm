# Author: Ronaldo Simeone <ronaldosimeon3@gmail.com>
# Date  : 2025-05-18

import requests
import base64
from bs4 import BeautifulSoup
from PIL import Image
import io
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict

# ========== CONFIGURAÇÕES ==========
TOKEN = "AXUR_TOKEN"
HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0"
}
SCRAPING_URL = "SCRAPPING_URL"
API_INFER_URL = "API_INFER_URL"
API_SUBMIT_URL = "API_SUBMIT_URL"
MODEL_NAME = "microsoft-florence-2-larg"
PROMPT_TAG = "<DETAILED_CAPTION>"
IMAGE_FILENAME = "imagem_scrape.png"


@dataclass
class ImageURL:
    """Representa o objeto `image_url` exigido pela API OpenAI-like."""
    url: str

@dataclass
class Content:
    """Item do array `content` em uma mensagem da API.

    Atributos:
        type: Tipo do conteúdo – `"text"` ou `"image_url"`.
        text: Texto do prompt (obrigatório se `type=="text"`).
        image_url: Instância de :class:`ImageURL`
                   (obrigatória se `type=="image_url"`).
    """
    type: str
    text: str = ""
    image_url: ImageURL = None

@dataclass
class Message:
    """Mensagem enviada à API.

    Atributos:
        role: Papel da mensagem – normalmente `"user"`.
        content: Lista de dicts resultantes de
                 :pyfunc:`dataclasses.asdict` em :class:`Content`.
    """
    role: str
    content: list

@dataclass
class InferencePayload:
    """Payload completo para o endpoint de chat/completions.

    Atributos:
        model: Nome do modelo (ex. `"microsoft-florence-2-large"`).
        messages: Lista com apenas uma :class:`Message`, pois este
                  endpoint aceita o mesmo formato da API da OpenAI.
    """
    model: str
    messages: list

#===================== FUNÇÕES AUXILIARES =====================

def fetch_image_data(scraping_url: str) -> tuple[str, str]:
    """Extrai a imagem em Base64 da página alvo.

    Args:
        scraping_url: URL que contém a tag `<img>` com data URI.

    Returns:
        Tuple `(mime_type, base64_data)` onde:
            * mime_type  – ex.: `"image/png"`;
            * base64_data – **apenas** a string Base64 (sem prefixo).

    Raises:
        requests.HTTPError: Se a página não retornar HTTP 200.
        ValueError: Se não encontrar `<img>` ou o formato da data URI
                    não corresponder ao regex.
    """
    print("Fazendo scraping da imagem...")

    response = requests.get(scraping_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    img_tag = soup.find("img")

    if not img_tag or "src" not in img_tag.attrs:
        raise ValueError("Imagem não encontrada na página.")

    data_url = img_tag["src"]
    match = re.match(r'data:(image/\w+);base64,(.+)', data_url)

    if not match:
        raise ValueError("Formato da imagem base64 inválido.")

    mime_type = match.group(1)
    base64_data = match.group(2)
    return mime_type, base64_data


def save_image(base64_data: str, filename: str) -> None:
    """Decodifica o Base64 e salva a imagem em disco (opcional).

    Args:
        base64_data: String em Base64 sem cabeçalho data URI.
        filename: Caminho onde a imagem será gravada.
    """
    image_data = base64.b64decode(base64_data)
    image = Image.open(io.BytesIO(image_data))
    image_format = image.format.lower()
    image.save(filename)
    print(f"Imagem salva como: {filename} ({image_format})")


def build_inference_payload(model: str, mime_type: str, base64_data: str) -> Dict[str, Any]:
    """Monta o payload JSON esperado pela API de inferência.

    Args:
        model: Nome do modelo.
        mime_type: Tipo MIME da imagem (ex. `"image/png"`).
        base64_data: String Base64 da imagem.

    Returns:
        Dicionário pronto para ser enviado via `json=` no `requests.post`.
    """
    image_url = ImageURL(url=f"data:{mime_type};base64,{base64_data}")
    content_list = [
        Content(type="text", text=PROMPT_TAG),
        Content(type="image_url", image_url=image_url)
    ]
    message = Message(role="user", content=[asdict(c) for c in content_list])
    payload = InferencePayload(model=model, messages=[asdict(message)])
    return asdict(payload)


def send_inference_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Envia o payload ao modelo e retorna o JSON de resposta.

    Args:
        payload: Dicionário produzido por :func:`build_inference_payload`.

    Returns:
        Resposta JSON convertida para `dict`.

    Raises:
        requests.HTTPError: Se o endpoint retornar status HTTP ≠ 200.
    """
    print("Enviando imagem para inferencia...")
    response = requests.post(API_INFER_URL, json=payload, headers=HEADERS)
    response.raise_for_status()
    print("Inferencia recebida com sucesso")
    return response.json()


def submit_response(data: Dict[str, Any]) -> None:
    """Submete o JSON de inferência ao endpoint de correção.

    Args:
        data: JSON exatamente como recebido da inferência.

    Raises:
        requests.HTTPError: Se o endpoint de submissão falhar.
    """
    print("Enviando resposta...")
    response = requests.post(API_SUBMIT_URL, json=data, headers=HEADERS)
    response.raise_for_status()
    print("Enviado com sucesso")


#==================== FUNÇÃO MAIN =======================
def main():
    try:
        mime_type, base64_data = fetch_image_data(SCRAPING_URL)
        save_image(base64_data, IMAGE_FILENAME)
        payload = build_inference_payload(MODEL_NAME, mime_type, base64_data)
        infer_response = send_inference_request(payload)
        submit_response(infer_response)
    except Exception as e:
        print(f"[✘] Erro: {e}")


if __name__ == "__main__":
    main()
