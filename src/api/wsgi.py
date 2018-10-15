import logging.config
import connexion

logging.basicConfig(level=logging.INFO)
app = connexion.App(__name__)
app.add_api("api_spec_model.yml")
application = app.app

if __name__ == "__main__":
    app.run(port=8080)
