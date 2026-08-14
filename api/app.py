"""
Flask application for AI Stroke Risk Prediction.
"""

import os
import sys

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

# Project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from prediction.predict import _load_artifacts


# Frontend directory
_FRONTEND_DIR = os.path.join(_PROJECT_ROOT, "frontend")


def create_app():
    app = Flask(
        __name__,
        static_folder=_FRONTEND_DIR,
        static_url_path=""
    )

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Load model once when the server starts
    model_loaded = False

    try:
        _load_artifacts()
        model_loaded = True
        app.logger.info("Model loaded successfully.")
    except Exception as exc:
        app.logger.error(
            "Failed to load model artifacts: %s",
            exc
        )

    app.config["MODEL_LOADED"] = model_loaded

    # Register API routes
    from api.routes import api_bp
    app.register_blueprint(api_bp)

    # -----------------------------
    # Frontend
    # -----------------------------

    @app.route("/")
    def home():
        return send_from_directory(
            _FRONTEND_DIR,
            "index.html"
        )

    # -----------------------------
    # Health check
    # -----------------------------

    @app.route("/api/health")
    def health():
        return jsonify({
            "status": "ok",
            "service": "stroke-risk-prediction-api",
            "model_loaded": app.config["MODEL_LOADED"]
        })

    # -----------------------------
    # Error handlers
    # -----------------------------

    @app.errorhandler(400)
    def bad_request(e):
        return jsonify({
            "success": False,
            "error": {
                "code": "BAD_REQUEST",
                "message": "The request could not be understood."
            }
        }), 400

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({
            "success": False,
            "error": {
                "code": "NOT_FOUND",
                "message": "The requested resource was not found."
            }
        }), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({
            "success": False,
            "error": {
                "code": "METHOD_NOT_ALLOWED",
                "message": "This HTTP method is not allowed."
            }
        }), 405

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({
            "success": False,
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected server error occurred."
            }
        }), 500

    return app


# Important for Gunicorn
app = create_app()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )