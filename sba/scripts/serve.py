"""Launch the static research demo on loopback only."""
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from functools import partial
from pathlib import Path
import webbrowser,argparse
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description='Run the local-only SBA research demo')
parser.add_argument('--port',type=int,default=0)
parser.add_argument('--no-open',action='store_true')
args=parser.parse_args()
class LocalHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control','no-store')
        super().end_headers()
server=ThreadingHTTPServer(('127.0.0.1',args.port),partial(LocalHandler,directory=str(ROOT/'web/dist')))
url=f'http://127.0.0.1:{server.server_port}'
print(f'Demo: {url}\nStop: Ctrl+C',flush=True)
if not args.no_open:webbrowser.open(url)
try:server.serve_forever()
except KeyboardInterrupt:pass
finally:server.server_close()
