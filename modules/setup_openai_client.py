import httpx
import openai


def setup_openai_client(api_key, proxy_url, base_url):
    """Initialize OpenAI client"""
    try:
        transport = httpx.HTTPTransport(proxy=proxy_url)
        httpx_client = httpx.Client(transport=transport)

        client = openai.OpenAI(
            api_key=api_key,
            http_client=httpx_client,
            base_url=base_url
        )
        return client
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {str(e)}")
        raise