"""Run the loopback-only web console for the two-camera SO-101 workbench."""

from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlsplit

from swap_skill import Board, CHECKPOINT, FRAME_ROOT, SwapMission, authorize_text, resume_requested, stack_requested


from settings import BOARD_HOST, WEB_PORT
PORT = WEB_PORT
ORIGIN = f'http://127.0.0.1:{PORT}'
STATIC = Path(__file__).with_name('swap_console_ui')
TOKEN = secrets.token_urlsafe(32)


class Console:
    def __init__(self):
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.active = False
        self.events = []
        self.frames = {}
        self.cameras_online = False
        self.state = 'idle'
        self.message = '准备就绪'
        self.details = {}

    def cameras(self):
        while True:
            try:
                with Board() as board:
                    board.ensure_cameras()
                    with board.client.open_sftp() as sftp:
                        sftp.get_channel().settimeout(8)
                        stamps = {}
                        last_change = time.monotonic()
                        last_heartbeat = 0
                        last_frames = {}
                        while True:
                            now = time.monotonic()
                            if now - last_heartbeat > 5:
                                with sftp.open(f'{FRAME_ROOT}/heartbeat', 'w') as file:
                                    file.write(str(time.time()))
                                last_heartbeat = now
                            for view in ('fixed', 'wrist'):
                                try:
                                    with sftp.open(f'{FRAME_ROOT}/{view}.jpg', 'rb') as file:
                                        frame = file.read()
                                except FileNotFoundError:
                                    continue
                                if frame.startswith(b'\xff\xd8') and frame.endswith(b'\xff\xd9') and frame != last_frames.get(view):
                                    last_frames[view] = frame
                                    stamps[view] = now
                            online = len(stamps) == 2 and now - min(stamps.values()) < 5
                            if online:
                                last_change = now
                            with self.lock:
                                self.frames.update(last_frames)
                                self.cameras_online = online
                            if now - last_change > 12:
                                raise RuntimeError('Camera frames stopped updating')
                            time.sleep(0.28)
            except Exception as exc:
                logging.info('Camera reconnect: %s', exc)
                with self.lock:
                    self.cameras_online = False
                time.sleep(1.5)

    def publish(self, state, message, **details):
        with self.lock:
            self.state, self.message = state, message
            self.details.update(details)
            self.events.append({'time': datetime.now().strftime('%H:%M:%S'), 'state': state, 'message': message})
            self.events = self.events[-32:]
        logging.info('%s %s', state, message)

    def start(self, text):
        from settings import require_hardware_enabled
        require_hardware_enabled()
        if not resume_requested(text) and not stack_requested(text):
            authorize_text(text)
        with self.lock:
            if self.active:
                raise RuntimeError('已有任务在运行')
            if not self.cameras_online:
                raise RuntimeError('双路相机尚未就绪')
            self.active = True
            self.cancel.clear()
            self.events.clear()
            self.details = {'step': 0, 'instruction': text}
        self.publish('checking', '指令已接收，开始检查工作台')
        threading.Thread(target=self.run, args=(text,), daemon=False).start()

    def run(self, text):
        try:
            with Board() as board:
                SwapMission(board, self.publish, self.cancel.is_set).run(text)
        except Exception as exc:
            logging.exception('Swap stopped')
            self.publish('stopped', str(exc) or '运行失败')
        finally:
            with self.lock:
                self.active = False

    def snapshot(self):
        with self.lock:
            return {'token': TOKEN, 'status': self.state, 'message': self.message, 'active': self.active,
                    'cameras_online': self.cameras_online, 'events': list(self.events),
                    'has_checkpoint': CHECKPOINT.exists(), **self.details}


console = Console()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, payload, kind, code=200):
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' blob:; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def json(self, value, code=200):
        self.send(json.dumps(value, ensure_ascii=False).encode(), 'application/json; charset=utf-8', code)

    def local(self):
        return self.headers.get('Host') == f'127.0.0.1:{PORT}'

    def do_GET(self):
        if not self.local():
            self.json({'error': '仅接受本机访问'}, 403)
            return
        route = urlsplit(self.path).path
        assets = {'/': ('index.html', 'text/html'), '/style.css': ('style.css', 'text/css'),
                  '/app.js': ('app.js', 'text/javascript')}
        if route in assets:
            name, mime = assets[route]
            self.send((STATIC / name).read_bytes(), mime + '; charset=utf-8')
        elif route == '/api/status':
            self.json(console.snapshot())
        elif route in ('/camera/fixed.jpg', '/camera/wrist.jpg'):
            view = route.split('/')[-1].split('.')[0]
            with console.lock:
                frame = console.frames.get(view)
            if frame:
                self.send(frame, 'image/jpeg')
            else:
                self.json({'error': '等待相机'}, 503)
        else:
            self.json({'error': '未找到'}, 404)

    def do_POST(self):
        if (not self.local() or self.headers.get('Origin') not in (None, ORIGIN)
                or not secrets.compare_digest(self.headers.get('X-Console-Token', ''), TOKEN)):
            self.json({'error': '本地指令校验失败'}, 403)
            return
        try:
            if self.path == '/api/run':
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 4096 or self.headers.get_content_type() != 'application/json':
                    raise ValueError('指令格式错误')
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict) or not isinstance(body.get('instruction'), str):
                    raise ValueError('请输入文字指令')
                console.start(body['instruction'])
            elif self.path == '/api/stop':
                console.cancel.set()
            else:
                self.json({'error': '未找到'}, 404)
                return
            self.json({'accepted': True}, 202)
        except (ValueError, RuntimeError) as exc:
            self.json({'error': str(exc)}, 409)


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if BOARD_HOST:
        threading.Thread(target=console.cameras, daemon=True).start()
    with ThreadingHTTPServer(('127.0.0.1', PORT), Handler) as server:
        logging.info('SO-101 console %s', ORIGIN)
        server.serve_forever()


if __name__ == '__main__':
    main()
