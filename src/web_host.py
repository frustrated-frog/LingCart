"""生产前端托管与 HTML 导航协商；JSON API 的会话权限保持独立。"""
from pathlib import Path

from fastapi.responses import FileResponse


def _accepts_html(header: str) -> bool:
    for media_range in header.lower().split(','):
        media, *parameters = media_range.strip().split(';')
        if media.strip() != 'text/html':
            continue
        quality = next((value.strip() for parameter in parameters
                        for key, separator, value in [parameter.strip().partition('=')]
                        if separator and key == 'q'), '1')
        try:
            return 0 < float(quality) <= 1
        except ValueError:
            return False
    return False


def register_frontend(app, dist: Path):
    if not dist.is_dir():
        return

    def document():
        return FileResponse(dist / 'index.html', headers={'Cache-Control': 'no-cache'})

    @app.middleware('http')
    async def run_inspector_navigation(request, call_next):
        parts = request.url.path.strip('/').split('/')
        if (request.method == 'GET' and len(parts) == 2 and parts[0] == 'runs' and parts[1]
                and 'session_id' not in request.query_params and _accepts_html(request.headers.get('accept', ''))):
            return document()
        return await call_next(request)

    @app.get('/{frontend_path:path}', include_in_schema=False)
    def frontend_app(frontend_path: str):
        requested = (dist / frontend_path).resolve()
        if dist.resolve() in requested.parents and requested.is_file() and requested != (dist / 'index.html').resolve():
            return FileResponse(requested)
        return document()
