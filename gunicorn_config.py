from os import getenv

worker_class = getenv("GUNICORN_WORKER_CLASS", "eventlet")

workers = int( getenv("GUNICORN_NUM_WORKERS", "1") )

bind = f"{ getenv( 'BIND_ADDRESS', '127.0.0.1' ) }:{ getenv('BIND_PORT', '6000') }"

accesslog = "-"  # Log access to stdout
errorlog = "-"   # Log errors to stderr
loglevel = getenv("GUNICORN_LOG_LEVEL", "info")

keepalive = int(getenv("GUNICORN_KEEPALIVE", "5"))
timeout = int(getenv("GUNICORN_TIMEOUT", "30"))
