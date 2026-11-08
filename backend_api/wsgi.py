"""WSGI entrypoint for production servers."""

from backend_api.app import app

application = app
