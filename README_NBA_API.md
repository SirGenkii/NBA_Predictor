# How to avoid timeout errors when using the NBA API ?


## Add proxy

Go to https://dashboard.webshare.io and create free rotating proxy.

Find your nba_api library in your virtual environment, for example:
    
```
    E:\Documents_\Dev\NBA_Predictor\.venv\Lib\site-packages\nba_api\library\debug\debug.py
    
```

## Update nba_api files

Add this line to .venv\Lib\site-packages\nba_api\library\debug\debug.py : 

```python


def build_webshare_rotating_residential(host, port, username, password, n_start=30, n_end=80):
    """
    Build a rotating residential proxy URL from the given parameters.
    Return formatted proxy URL : "http://username:password@host:port"
    """
    
    urls = []
    
    for i in range(n_start, n_end + 1):
        urls.append(f"http://{username}-{i}:{password}@{host}:{port}")
    
    return urls
    
PROXY = build_webshare_rotating_residential(
    host="p.webshare.io",   
    port=80,
    username="tklpflkn",
    password="ekjqv341wl4w",
    n_start=30,
    n_end=1300
)
```

Replace this method in .venv\Lib\site-packages\nba_api\library\http.py:

```
    @classmethod
    def get_session(cls):
        session = cls._session
        if session is None:
            session = requests.Session()
            cls._session = session
        return session
        ```

with 

```
    @classmethod
    def get_session(cls):
        # Toujours renvoyer une nouvelle session pour forcer la rotation d’IP
        session = requests.Session()
        cls._session = session
        return session
```