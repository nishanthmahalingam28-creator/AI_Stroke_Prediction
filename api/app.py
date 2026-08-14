"""
api/app.py
------------------------
STEP 8 - Flask application factory for the Stroke Risk Prediction API.

Loads the saved Step 7 model artifacts ONCE at startup (not per-request,
not retrained). Registers the /api routes, CORS, and JSON error handlers
for 400/404/405/500 so no Python stack traces are ever exposed to a
client.
"""

import os
import sys

from flask import Flask, jsonify
from flask_cors import CORS

# Make sure the project root (parent of api/) is importable so that
# `from prediction.predict import ...` works regardless of cwd.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from prediction.predict import _load_artifacts  # noqa: E402


def create_app():
    app = Flask(__name__)

    # Development-friendly CORS config so a future frontend (likely on
    # a different local port, e.g. React/Vite on :5173 or :3000) can
    # call this API. Not locked to a specific origin yet since no
    # frontend exists; this is deliberately dev-only.
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # ------------------------------------------------------------
    # Load the saved model/preprocessor/config ONCE at startup.
    # This does NOT retrain anything -- it only deserializes the
    # already-trained Step 7 artifacts from disk.
    # ------------------------------------------------------------
    model_loaded = False
    try:
        _load_artifacts()
        model_loaded = True
    except Exception as exc:  # noqa: BLE001
        # Don't crash the whole app import; /api/health should still
        # be able to report model_loaded: false instead of the server
        # failing to start silently with no explanation.
        app.logger.error("Failed to load model artifacts at startup: %s", exc)

    app.config["MODEL_LOADED"] = model_loaded

    from api.routes import api_bp

    app.register_blueprint(api_bp)

    # ------------------------------------------------------------
    # JSON error handlers -- never leak stack traces to the client.
    # ------------------------------------------------------------
    @app.errorhandler(400)
    def bad_request(e):
        return (
            jsonify(
                {
                    "success": False,
                    "error": {
                        "code": "BAD_REQUEST",
                        "message": "The request could not be understood.",
                    },
                }
            ),
            400,
        )

    @app.errorhandler(404)
    def not_found(e):
        return (
            jsonify(
                {
                    "success": False,
                    "error": {
                        "code": "NOT_FOUND",
                        "message": "The requested resource was not found.",
                    },
                }
            ),
            404,
        )

    @app.errorhandler(405)
    def method_not_allowed(e):
        return (
            jsonify(
                {
                    "success": False,
                    "error": {
                        "code": "METHOD_NOT_ALLOWED",
                        "message": "This HTTP method is not allowed for this endpoint.",
                    },
                }
            ),
            405,
        )

    @app.errorhandler(500)
    def internal_error(e):
        return (
            jsonify(
                {
                    "success": False,
                    "error": {
                        "code": "INTERNAL_SERVER_ERROR",
                        "message": "An unexpected server error occurred.",
                    },
                }
            ),
            500,
        )

    # Catch-all for any other unhandled exception -- guarantees no
    # stack trace / internal path ever reaches the client, even for
    # errors Flask's default handlers wouldn't otherwise catch.
    @app.errorhandler(Exception)
    def unhandled_exception(e):
        app.logger.exception("Unhandled exception")
        return (
            jsonify(
                {
                    "success": False,
                    "error": {
                        "code": "INTERNAL_SERVER_ERROR",
                        "message": "An unexpected server error occurred.",
                    },
                }
            ),
            500,
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=False)
