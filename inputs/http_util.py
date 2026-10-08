"""Small HTTP helper shared by the data-layer scripts (standard library only).

Python from python.org on macOS ships without CA certificates, so when the
default trust store is empty the system bundle /etc/ssl/cert.pem is used.
"""

import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

# Identifies this project to the servers it queries. Set COLABFOLD_CONTACT
# (for example an e-mail address) to add a contact, as the ColabFold MSA
# server asks.
USER_AGENT = 'alphafold-on-tpu/0.1 (+https://github.com/lorenzopazienza/alphafold-on-tpu)'
if os.environ.get('COLABFOLD_CONTACT'):
  USER_AGENT += f' {os.environ["COLABFOLD_CONTACT"]}'


def _ssl_context():
  ctx = ssl.create_default_context()
  if not ctx.cert_store_stats().get('x509_ca') and os.path.exists('/etc/ssl/cert.pem'):
    ctx = ssl.create_default_context(cafile='/etc/ssl/cert.pem')
  return ctx


SSL_CONTEXT = _ssl_context()


def request(url, data=None, timeout=60):
  """One HTTP request. Returns (status, headers, body); raises on network errors.

  data, if given, is sent as an application/x-www-form-urlencoded POST.
  """
  body = urllib.parse.urlencode(data).encode() if data is not None else None
  req = urllib.request.Request(url, data=body, headers={'User-Agent': USER_AGENT})
  try:
    with urllib.request.urlopen(req, timeout=timeout, context=SSL_CONTEXT) as r:
      return r.status, dict(r.headers), r.read()
  except urllib.error.HTTPError as e:
    return e.code, dict(e.headers or {}), e.read()


def fetch(url, attempts=5, timeout=120):
  """GET with retries on network errors and 5xx; raises on 4xx."""
  for attempt in range(1, attempts + 1):
    try:
      status, _, body = request(url, timeout=timeout)
      if status == 200:
        return body
      if status < 500:
        raise RuntimeError(f'HTTP {status} for {url}')
      problem = f'HTTP {status}'
    except (urllib.error.URLError, OSError) as e:
      if isinstance(e, urllib.error.URLError) and isinstance(e.reason, ssl.SSLError):
        raise
      problem = str(e)
    if attempt == attempts:
      raise RuntimeError(f'{url}: {problem} after {attempts} attempts')
    print(f'  retry {attempt} for {url}: {problem}', flush=True)
    time.sleep(5 * attempt)
