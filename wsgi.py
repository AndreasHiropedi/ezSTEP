import os

os.environ['NUMBA_CACHE_DIR'] = '/tmp'

import sys
sys.path.insert(0, '/var/www/wsgi/ezSTEP/app')

from main_page import server as application

