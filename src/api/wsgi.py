import logging.config
import utils
import connexion

_LOG = logging.getLogger()


logging.basicConfig(level=logging.INFO)
app = connexion.App(__name__)
app.add_api('api_spec_model.yml')
application = app.app


if __name__ == '__main__':
    # run our standalone gevent server
    app.run(port=8080)