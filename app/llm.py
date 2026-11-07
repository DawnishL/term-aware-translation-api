import httpx
from app.config import Settings

class OllamaClient:
    def __init__(self, settings: Settings, timeout: float = 120) -> None:
        self.url = settings.ollama_url.rstrip("/") + "/api/chat"
        self.model = settings.ollama_model
        self.timeout = timeout
    def chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "stream": False,
            "options": {"temperature": 0},
        }
        
        response = httpx.post(self.url, json=payload, timeout=self.timeout)
        response.raise_for_status

        data = response.json()
        return data["message"]["content"].strip()
    
if __name__ == "__main__":
    from app.config import get_settings

    client = OllamaClient(get_settings())
    reply = client.chat(
        system="你是一名专业的英译中技术翻译。只输出译文，不要解释。",
        user="Check the battery level indicator before using the power adapter.",
    )
    print(reply)