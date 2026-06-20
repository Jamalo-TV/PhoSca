import logging

import structlog


def configure_logging(log_level: str) -> None:
    logging.basicConfig(level=log_level.upper(), format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(log_level.upper())),
        cache_logger_on_first_use=True,
    )

