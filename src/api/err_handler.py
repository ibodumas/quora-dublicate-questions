import utils
from werkzeug.exceptions import HTTPException, HTTP_STATUS_CODES
from flask import jsonify
import logging
import traceback


LOG = logging.getLogger(utils.NAME)

CUSTOM_MESSAGES = {500: "An unknown error, contact support."}


def _err_endpoint(error):
    j = jsonify(error.description)
    j.status_code = error.code
    return j


def _application_err(error):
    if utils.DEBUG:
        te = traceback.TracebackException.from_exception(error)
        lines = list(te.format())
        print("".join(lines), flush=True)

    if not isinstance(error, HTTPException):
        error.response = None
        error.code = 500
        if utils.DEBUG:
            error.description = "{}: {}".format(type(error).__name__, str(error))
        else:
            error.description = CUSTOM_MESSAGES[500]

    elif error.code in CUSTOM_MESSAGES:
        error.description = CUSTOM_MESSAGES[error.code]

    return _err_endpoint(error)


def add_handler(app):
    app.add_error_handler(Exception, _application_err)

    for code in utils.STANDARD_ERRORS:
        app.add_error_handler(code, _err_endpoint)

    all_errors = set(HTTP_STATUS_CODES)
    remaining_errors = all_errors.difference(set(utils.STANDARD_ERRORS))

    for code in remaining_errors:
        try:
            app.add_error_handler(code, _application_err)
        except KeyError:
            pass

    return app