import logging
import os


LOG_DIRECTORY = "logs"
LOG_FILE = os.path.join(
    LOG_DIRECTORY,
    "app.log"
)


def setup_logger():

    os.makedirs(
        LOG_DIRECTORY,
        exist_ok=True
    )

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - "
            "%(levelname)s - "
            "%(message)s"
        ),
        handlers=[
            logging.FileHandler(
                LOG_FILE,
                encoding="utf-8"
            ),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger("diabetes_ai")